# baseline
from .GPD import build_gpd
from .PointNetGPD import build_pointnetgpd

# our methods
from .PointNet2 import build_pointnet2
from .PointNet2_local import build_pointnet2_local
from .PointNet2_tcls import build_pointnet2_cls
from .EdgePointNet2Down import build_edgepointnet2down
from .EdgePointNet2DownUp import build_edgepointnet2downup


def build_model(cfg):
    if cfg.MODEL.TYPE == "GPD":
        net, loss_func, metric = build_gpd(cfg)
    elif cfg.MODEL.TYPE == "PointNetGPD":
        net, loss_func, metric = build_pointnetgpd(cfg)
    elif cfg.MODEL.TYPE == "PN2":
        net, loss_func, metric = build_pointnet2(cfg)
    elif cfg.MODEL.TYPE == "PN2_CLS":
        net, loss_func, metric = build_pointnet2_cls(cfg)
    elif cfg.MODEL.TYPE == "PN2_LOCAL":
        net, loss_func, metric = build_pointnet2_local(cfg)
    elif cfg.MODEL.TYPE == "EDGEPN2D":
        net, loss_func, metric = build_edgepointnet2down(cfg)
    elif cfg.MODEL.TYPE == "EDGEPN2DU":
        net, loss_func, metric = build_edgepointnet2downup(cfg)
    else:
        raise ValueError("Unknown model: {}.".format(cfg.MODEL.MODEL))

    return net, loss_func, metric
