import torch
from utils.general_utils import strip_symmetric, build_scaling_rotation
import math

class GroundModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = torch.logit

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, model_args):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.restore(model_args)
        self.setup_functions()

    def restore(self, model_args):
        (self.active_sh_degree,
        self._xyz,
        # self._y,
        # self._z,
        self._features_dc,
        self._features_rest,
        self._feats3D,
        self._scaling,
        self._rotation,
        self._opacity) = model_args

    @property
    def get_scaling(self):
        scale_y = torch.ones_like(self._xyz[:, 0]) * math.log(0.001)
        scaling = torch.stack((self._scaling[:, 0], scale_y, self._scaling[:, 1]), dim=1).cuda()
        # scaling = torch.stack((self._scaling, scale_y, self._scaling), dim=1).cuda()
        return self.scaling_activation(scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_3D_features(self):
        return torch.softmax(self._feats3D, dim=-1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
