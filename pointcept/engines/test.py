import gc
import os
import math
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel

from .defaults import create_ddp_model
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)


TESTERS = Registry("testers")


class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        self.split_aspect = cfg.get("max_test_aspect", float("inf"))
        self.split_spacing = cfg.get("split_spacing", self.split_aspect - 1)
        self.spa_native_mode = cfg.get("spa_native_mode", None)
        
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def get_base_model(self):
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module
        else:
            return self.model

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                # FIXME: temp solution to fit mmcv's spconv auto conversion
                #        only works for spconv 3d
                if len(value.shape) == 5 and not key.endswith("table"):   
                    value = value.permute(1, 2, 3, 4, 0)
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        # enable spa native impl
        from pointcept.models.sparse_proxy_ptv3.sparse_attention import SparseAttention
        for name, module in model.named_modules():
            if isinstance(module, SparseAttention):
                if self.spa_native_mode == "taichi":
                    module.use_taichi = True
                    module.use_warp = False
                    self.logger.info(f"Setting {name} to use taichi native mode.")
                if self.spa_native_mode == "warp":
                    module.use_taichi = False
                    module.use_warp = True
                    self.logger.info(f"Setting {name} to use warp native mode.")
        return model

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.batch_size_test_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)


@TESTERS.register_module()
class SemSegTester(TesterBase):
    @torch.no_grad()
    def test(self):
        vis_cnt = int(os.environ.get("SPA_VIS_CNT", "0"))
        vis_tgt = os.environ.get("SPA_VIS_TARGET_SCENE", None)
        is_vis = vis_cnt > 0
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        logger.info(f"vis mode: {is_vis}")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        # create submit folder only on main process
        if not is_vis:
            if (
                self.cfg.data.test.type == "ScanNetDataset"
                or self.cfg.data.test.type == "ScanNet200Dataset"
            ) and comm.is_main_process():
                make_dirs(os.path.join(save_path, "submit"))
            elif (
                self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
            ):
                make_dirs(os.path.join(save_path, "submit"))
            elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
                import json

                make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
                make_dirs(os.path.join(save_path, "submit", "test"))
                submission = dict(
                    meta=dict(
                        use_camera=False,
                        use_lidar=True,
                        use_radar=False,
                        use_map=False,
                        use_external=False,
                    )
                )
                with open(
                    os.path.join(save_path, "submit", "test", "submission.json"), "w"
                ) as f:
                    json.dump(submission, f, indent=4)
        comm.synchronize()
        record = {}
        test_dataset = self.test_loader.dataset
        # fragment inference
        sample_time = []
        total_iter = 0
        for idx, data_dict in enumerate(self.test_loader):
            if is_vis and idx >= vis_cnt and vis_tgt is None:
                raise RuntimeError("exiting vis")
            end = time.perf_counter()
            data_dict = data_dict[0]  # current assume batch size is 1
            if vis_tgt is not None and vis_tgt != data_dict["name"]:
                logger.info(f"skip {data_dict['name']}: not vis target ({vis_tgt})")
                continue
            fragment_list = data_dict.pop("fragment_list")
            fragment_iter_idx = data_dict.get("fragment_iter", None)
            num_aug = data_dict.get("num_aug", None)
            segment = data_dict.pop("segment")
            # test as evaluate
            data_name = data_dict.pop("name")
            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
            if os.path.isfile(pred_save_path):
                logger.info(
                    "{}/{}: {}, loaded pred and label.".format(
                        idx + 1, len(self.test_loader), data_name
                    )
                )
                pred = np.load(pred_save_path)
            else:
                pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
                if fragment_iter_idx is not None:
                    fragment_iter = test_dataset.get_test_generator(fragment_iter_idx)
                    data, i_aug = next(fragment_iter)
                    i = 0
                    while data is not None:
                        # split alonge the longest axis
                        pc_min = data["coord"].min(dim=0).values
                        pc_max = data["coord"].max(dim=0).values
                        pc_range = pc_max - pc_min
                        max_axis = pc_range.argmax()
                        max_axis_size = pc_range.max()
                        min_axis_size = pc_range.min()
                        pc_aspect = max_axis_size / min_axis_size
                        if pc_aspect <= self.split_aspect:
                            data_list = [data]
                        else:
                            num_splits = math.ceil((pc_aspect - self.split_aspect) / self.split_spacing) + 1
                            split_extent = self.split_aspect * 0.5 * min_axis_size + 1e-6
                            split_spacing = (pc_aspect - self.split_aspect) / (num_splits - 1)
                            data_list = []
                            split_start = pc_min[max_axis] + self.split_aspect * 0.5 * min_axis_size
                            for s in range(num_splits):
                                center = split_start + split_spacing * s
                                mask =   (data["coord"][:, max_axis] >= center - split_extent) \
                                       & (data["coord"][:, max_axis] <= center + split_extent)
                                data_split = data.copy()
                                for key in data_split.keys():
                                    try:
                                        if len(data_split[key]) == data["coord"].shape[0]:
                                            data_split[key] = data_split[key][mask]
                                    except:
                                        pass
                                center_coord = data_split["coord"].mean(dim=0)
                                data_split["coord"] -= center_coord
                                min_grid_coord = data_split["grid_coord"].min(dim=0).values
                                data_split["grid_coord"] -= min_grid_coord
                                data_split["offset"] = data_split["offset"].new_tensor([mask.sum()])
                                data_list.append(data_split)
                        for data in data_list:
                            torch.cuda.empty_cache()
                            input_dict = collate_fn([data])
                            for key in input_dict.keys():
                                if isinstance(input_dict[key], torch.Tensor):
                                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
                            idx_part = input_dict["index"]
                            t0 = time.perf_counter()
                            with torch.no_grad():
                                pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                                pred_part = F.softmax(pred_part, -1)
                            t1 = time.perf_counter()
                            sample_time.append(t1 - t0)
                            total_iter += 1
                            bs = 0
                            for be in input_dict["offset"]:
                                pred[idx_part[bs:be], :] += pred_part[bs:be]
                                bs = be
                        logger.info(
                            "Test: {}/{}-{data_name}, Iter: {iter:04d}, Aug: {batch_idx}/{batch_num}, Input Size: {input_size}".format(
                                idx + 1,
                                len(self.test_loader),
                                iter=i+1,
                                data_name=data_name,
                                batch_idx=i_aug,
                                batch_num=num_aug,
                                input_size=data['coord'].shape[0]
                            )
                        )
                        i += 1
                        try:
                            data, i_aug = next(fragment_iter)
                        except StopIteration:
                            data = None
                else:
                    for i in range(len(fragment_list)):
                        data = fragment_list[i]
                        # split alonge the longest axis
                        pc_min = data["coord"].min(dim=0).values
                        pc_max = data["coord"].max(dim=0).values
                        pc_range = pc_max - pc_min
                        max_axis = pc_range.argmax()
                        max_axis_size = pc_range.max()
                        min_axis_size = pc_range.min()
                        pc_aspect = max_axis_size / min_axis_size
                        if pc_aspect <= self.split_aspect:
                            data_list = [data]
                        else:
                            num_splits = math.ceil((pc_aspect - self.split_aspect) / self.split_spacing) + 1
                            split_extent = self.split_aspect * 0.5 * min_axis_size + 1e-6
                            split_spacing = (pc_aspect - self.split_aspect) / (num_splits - 1)
                            data_list = []
                            split_start = pc_min[max_axis] + self.split_aspect * 0.5 * min_axis_size
                            for s in range(num_splits):
                                center = split_start + split_spacing * s
                                mask =   (data["coord"][:, max_axis] >= center - split_extent) \
                                       & (data["coord"][:, max_axis] <= center + split_extent)
                                data_split = data.copy()
                                for key in data_split.keys():
                                    try:
                                        if len(data_split[key]) == data["coord"].shape[0]:
                                            data_split[key] = data_split[key][mask]
                                    except:
                                        pass
                                center_coord = data_split["coord"].mean(dim=0)
                                data_split["coord"] -= center_coord
                                min_grid_coord = data_split["grid_coord"].min(dim=0).values
                                data_split["grid_coord"] -= min_grid_coord
                                data_split["offset"] = data_split["offset"].new_tensor([mask.sum()])
                                data_list.append(data_split)
                        for data in data_list:
                            torch.cuda.empty_cache()
                            input_dict = collate_fn([data])
                            for key in input_dict.keys():
                                if isinstance(input_dict[key], torch.Tensor):
                                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
                            idx_part = input_dict["index"]
                            t0 = time.perf_counter()
                            with torch.no_grad():
                                pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                                pred_part = F.softmax(pred_part, -1)
                            t1 = time.perf_counter()
                            sample_time.append(t1 - t0)
                            total_iter += 1
                            with torch.no_grad():
                                bs = 0
                                for be in input_dict["offset"]:
                                    pred[idx_part[bs:be], :] += pred_part[bs:be]
                                    bs = be
                        logger.info(
                            "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                                idx + 1,
                                len(self.test_loader),
                                data_name=data_name,
                                batch_idx=i,
                                batch_num=len(fragment_list),
                            )
                        )
                pred = pred.max(1)[1].data.cpu().numpy()
                if not is_vis:
                    np.save(pred_save_path, pred)
            if "origin_segment" in data_dict.keys():
                assert "inverse" in data_dict.keys()
                pred = pred[data_dict["inverse"]]
                segment = data_dict["origin_segment"]
            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))
            
            if is_vis:
                raw_data = test_dataset.get_data(fragment_iter_idx)
                base_model = self.get_base_model()
                debug_states = getattr(base_model, "debug_states", None)
                info = dict(
                    model=base_model, input=input_dict,
                    raw_data=raw_data, pred=pred, target=segment,
                    backbone_states=debug_states,
                    metrics=dict(iou=iou, acc=acc, m_iou=m_iou, m_acc=m_acc),
                )
                torch.save(info, os.path.join(save_path, f"vis_{data_name}.pth"))

            batch_time.update(time.perf_counter() - end)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )
            if not is_vis and (
                self.cfg.data.test.type == "ScanNetDataset"
                or self.cfg.data.test.type == "ScanNet200Dataset"
            ):
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                    fmt="%d",
                )
            elif not is_vis and self.cfg.data.test.type == "SemanticKITTIDataset":
                # 00_000000 -> 00, 000000
                sequence_name, frame_name = data_name.split("_")
                os.makedirs(
                    os.path.join(
                        save_path, "submit", "sequences", sequence_name, "predictions"
                    ),
                    exist_ok=True,
                )
                pred = pred.astype(np.uint32)
                pred = np.vectorize(
                    self.test_loader.dataset.learning_map_inv.__getitem__
                )(pred).astype(np.uint32)
                pred.tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "sequences",
                        sequence_name,
                        "predictions",
                        f"{frame_name}.label",
                    )
                )
            elif not is_vis and self.cfg.data.test.type == "NuScenesDataset":
                np.array(pred + 1).astype(np.uint8).tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "lidarseg",
                        "test",
                        "{}_lidarseg.bin".format(data_name),
                    )
                )

        if total_iter > 1000:
            sample_time.sort()
            sample_time = sample_time[int(total_iter * 0.1):int(total_iter * 0.9)]
            avg_time = sum(sample_time) / len(sample_time)
            logger.info(f"rank {comm.get_local_rank()}: avg inference time {avg_time * 1000}ms")
        
        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class ClsTester(TesterBase):
    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        for i, input_dict in enumerate(self.test_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            end = time.perf_counter()
            with torch.no_grad():
                output_dict = self.model(input_dict)
            output = output_dict["cls_logits"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred, label, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            intersection_meter.update(intersection), union_meter.update(
                union
            ), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.perf_counter() - end)

            logger.info(
                "Test: [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {accuracy:.4f} ".format(
                    i + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    accuracy=accuracy,
                )
            )

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU, mAcc, allAcc
            )
        )

        for i in range(self.cfg.data.num_classes):
            logger.info(
                "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=accuracy_class[i],
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class PartSegTester(TesterBase):
    def test(self):
        test_dataset = self.test_loader.dataset
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()

        num_categories = len(self.test_loader.dataset.categories)
        iou_category, iou_count = np.zeros(num_categories), np.zeros(num_categories)
        self.model.eval()

        save_path = os.path.join(
            self.cfg.save_path, "result", "test_epoch{}".format(self.cfg.test_epoch)
        )
        make_dirs(save_path)

        for idx in range(len(test_dataset)):
            end = time.perf_counter()
            data_name = test_dataset.get_data_name(idx)

            data_dict_list, label = test_dataset[idx]
            pred = torch.zeros((label.size, self.cfg.data.num_classes)).cuda()
            batch_num = int(np.ceil(len(data_dict_list) / self.cfg.batch_size_test))
            for i in range(batch_num):
                s_i, e_i = i * self.cfg.batch_size_test, min(
                    (i + 1) * self.cfg.batch_size_test, len(data_dict_list)
                )
                input_dict = collate_fn(data_dict_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad():
                    pred_part = self.model(input_dict)["cls_logits"]
                    pred_part = F.softmax(pred_part, -1)
                if self.cfg.empty_cache:
                    torch.cuda.empty_cache()
                pred_part = pred_part.reshape(-1, label.size, self.cfg.data.num_classes)
                pred = pred + pred_part.total(dim=0)
                logger.info(
                    "Test: {} {}/{}, Batch: {batch_idx}/{batch_num}".format(
                        data_name,
                        idx + 1,
                        len(test_dataset),
                        batch_idx=i,
                        batch_num=batch_num,
                    )
                )
            pred = pred.max(1)[1].data.cpu().numpy()

            category_index = data_dict_list[0]["cls_token"]
            category = self.test_loader.dataset.categories[category_index]
            parts_idx = self.test_loader.dataset.category2part[category]
            parts_iou = np.zeros(len(parts_idx))
            for j, part in enumerate(parts_idx):
                if (np.sum(label == part) == 0) and (np.sum(pred == part) == 0):
                    parts_iou[j] = 1.0
                else:
                    i = (label == part) & (pred == part)
                    u = (label == part) | (pred == part)
                    parts_iou[j] = np.sum(i) / (np.sum(u) + 1e-10)
            iou_category[category_index] += parts_iou.mean()
            iou_count[category_index] += 1

            batch_time.update(time.perf_counter() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} "
                "({batch_time.avg:.3f}) ".format(
                    data_name, idx + 1, len(self.test_loader), batch_time=batch_time
                )
            )

        ins_mIoU = iou_category.sum() / (iou_count.sum() + 1e-10)
        cat_mIoU = (iou_category / (iou_count + 1e-10)).mean()
        logger.info(
            "Val result: ins.mIoU/cat.mIoU {:.4f}/{:.4f}.".format(ins_mIoU, cat_mIoU)
        )
        for i in range(num_categories):
            logger.info(
                "Class_{idx}-{name} Result: iou_cat/num_sample {iou_cat:.4f}/{iou_count:.4f}".format(
                    idx=i,
                    name=self.test_loader.dataset.categories[i],
                    iou_cat=iou_category[i] / (iou_count[i] + 1e-10),
                    iou_count=int(iou_count[i]),
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)
