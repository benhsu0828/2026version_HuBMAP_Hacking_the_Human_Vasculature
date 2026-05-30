# 匯入即註冊：把 InternImage backbone 註冊進 MMDet v3 的 MODELS registry。
# config 內以 `custom_imports = dict(imports=['models.intern_image'])` 觸發。
from .intern_image import InternImage  # noqa: F401

__all__ = ['InternImage']
