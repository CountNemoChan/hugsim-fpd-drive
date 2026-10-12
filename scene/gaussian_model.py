import torch
from utils.general_utils import strip_symmetric, build_scaling_rotation
import tinycudann as tcnn
from scene.ground_model import GroundModel

class GaussianModel:

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

    def __init__(self, sh_degree : int, feat_mutable=True, affine=False):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._feats3D = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.spatial_lr_scale = 0
        self.feat_mutable = feat_mutable
        self.setup_functions()

        self.pos_enc = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={"otype": "Frequency", "n_frequencies": 2},
        )
        self.dir_enc = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 3,
            },
        )

        self.affine = affine
        if affine:
            self.appearance_model = tcnn.Network(
                n_input_dims=self.pos_enc.n_output_dims + self.dir_enc.n_output_dims,
                n_output_dims=12,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 32,
                    "n_hidden_layers": 2,
                }
            )
        else:
            self.appearance_model = None

        self.ground_model = None

    def restore(self, model_args):
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._feats3D,
        self._scaling,
        self._rotation,
        self._opacity,
        self.spatial_lr_scale,
        appearance_state_dict,
        ground_model_params,
        ) = model_args
        self.appearance_model.load_state_dict(appearance_state_dict, strict=False)
        if ground_model_params is not None:
            self.ground_model = GroundModel(self.max_sh_degree, model_args=ground_model_params)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_full_scaling(self):
        assert self.ground_model is not None
        return torch.cat([self.scaling_activation(self._scaling), self.ground_model.get_scaling])

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_full_rotation(self):
        assert self.ground_model is not None
        return torch.cat([self.rotation_activation(self._rotation), self.ground_model.get_rotation])

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_full_xyz(self):
        assert self.ground_model is not None
        return torch.cat([self._xyz, self.ground_model.get_xyz])

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_full_features(self):
        assert self.ground_model is not None
        sh = torch.cat((self._features_dc, self._features_rest), dim=1)
        return torch.cat([sh, self.ground_model.get_features])

    @property
    def get_3D_features(self):
        return torch.softmax(self._feats3D, dim=-1)

    @property
    def get_full_3D_features(self):
        assert self.ground_model is not None
        return torch.cat([torch.softmax(self._feats3D, dim=-1), self.ground_model.get_3D_features])

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_full_opacity(self):
        assert self.ground_model is not None
        return torch.cat([self.opacity_activation(self._opacity), self.ground_model.get_opacity])

    # def get_covariance(self, scaling_modifier = 1):
    #     return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
