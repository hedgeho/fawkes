"""Vendored face-recognition backbone definitions.

Each module keeps the licence header of the project it was copied from and lists the
modifications made here (all of them are removals of training-only or GPU-only code paths
so the files import without timm/fvcore and run on a CPU-only torch build).

- ``iresnet``: IResNet from insightface ``recognition/arcface_torch/backbones/iresnet.py`` (MIT).
- ``adaface_ir``: IR-101 from mk-minchul/AdaFace ``net.py`` as shipped in CVLface (MIT).
- ``vit``: face ViT from insightface ``recognition/arcface_torch/backbones/vit.py`` (MIT),
  also the architecture of LVFace and of the CVLface ``models/vit`` variant.
"""
