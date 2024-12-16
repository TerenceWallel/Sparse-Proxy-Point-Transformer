_base_ = ["./semseg-sppt-6fold-T1.py"]

weight = "exp/s3dis/semseg-pt-v3m1-0-rpe-T5/model/model_best.pth"

data = dict(
    train=dict(
        split=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
    ),
    val=dict(
        split="Area_5",
    ),
    test=dict(
        split="Area_5",
    ),
)
