import torch
import torch.nn as nn

class EquivariantLayerNorm(nn.Module):
    
    def __init__(self, eps=1e-5):
        super().__init__()

        self.eps = eps

    def forward(self, node_input, L):
        # input shape: [atom, 2l+1, feature_num]
        input_norm = torch.norm(node_input,dim=1,keepdim=True)
        RMS = torch.sqrt(torch.mean(torch.square(input_norm),dim=-1,keepdim=True)+self.eps)#.detach()
        if L == 0:
            mean = torch.mean(torch.mean(node_input,dim=-1,keepdim=True),dim=-1,keepdim=True)#.detach()
            # normed = torch.div(node_input-mean.repeat(node_input.shape[1],node_input.shape[2],1).permute(2, 0, 1), 
            #                    RMS.repeat(node_input.shape[1],node_input.shape[2],1).permute(2, 0, 1)+1e-5)
            normed = (node_input-mean)/RMS
        else:
            # normed = torch.div(node_input, RMS.repeat(node_input.shape[1],node_input.shape[2],1).permute(2, 0, 1)+1e-5)
            normed = node_input/RMS
        return normed

class EquivariantLayerNormArray(nn.Module):
    def __init__(
        self,
        lmax: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        normalization: str = "component",
    ):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.affine_weight = nn.Parameter(
                torch.ones(lmax + 1, num_channels)
            )
            self.affine_bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in ["norm", "component"]
        self.normalization = normalization

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps})"

    @torch.amp.autocast('cuda',enabled=False)
    def forward(self, node_input):
        """
        Assume input is of shape [N, sphere_basis, C]
        """

        out = []

        for lval in range(self.lmax + 1):
            start_idx = lval**2
            length = 2 * lval + 1

            feature = node_input.narrow(1, start_idx, length)

            # For scalars, first compute and subtract the mean
            if lval == 0:
                feature_mean = torch.mean(feature, dim=2, keepdim=True)
                feature = feature - feature_mean

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == "norm":
                feature_norm = feature.pow(2).sum(
                    dim=1, keepdim=True
                )  # [N, 1, C]
            elif self.normalization == "component":
                feature_norm = feature.pow(2).mean(
                    dim=1, keepdim=True
                )  # [N, 1, C]

            feature_norm = torch.mean(
                feature_norm, dim=2, keepdim=True
            )  # [N, 1, 1]
            feature_norm = (feature_norm + self.eps).pow(-0.5)

            if self.affine:
                weight = self.affine_weight.narrow(0, lval, 1)  # [1, C]
                weight = weight.view(1, 1, -1)  # [1, 1, C]
                feature_norm = feature_norm * weight  # [N, 1, C]

            feature = feature * feature_norm

            if self.affine and lval == 0:
                bias = self.affine_bias
                bias = bias.view(1, 1, -1)
                feature = feature + bias

            out.append(feature)

        out = torch.cat(out, dim=1)

        return out


class EquivariantLayerNormArraySphericalHarmonics(nn.Module):
    """
    1. Normalize over L = 0.
    2. Normalize across all m components from degrees L > 0.
    3. Do not normalize separately for different L (L > 0).
    """

    def __init__(
        self,
        lmax: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        normalization: str = "component",
        std_balance_degrees: bool = True,
    ):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.std_balance_degrees = std_balance_degrees

        # for L = 0
        self.norm_l0 = torch.nn.LayerNorm(
            self.num_channels, eps=self.eps, elementwise_affine=self.affine
        )

        # for L > 0
        if self.affine:
            self.affine_weight = nn.Parameter(
                torch.ones(self.lmax, self.num_channels)
            )
        else:
            self.register_parameter("affine_weight", None)

        assert normalization in ["norm", "component"]
        self.normalization = normalization

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2 - 1, 1)
            for lval in range(1, self.lmax + 1):
                start_idx = lval**2 - 1
                length = 2 * lval + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = (
                    1.0 / length
                )
            balance_degree_weight = balance_degree_weight / self.lmax
            self.register_buffer(
                "balance_degree_weight", balance_degree_weight
            )
        else:
            self.balance_degree_weight = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, std_balance_degrees={self.std_balance_degrees})"

    @torch.amp.autocast('cuda',enabled=False)
    def forward(self, node_input):
        """
        Assume input is of shape [N, sphere_basis, C]
        """

        out = []

        # for L = 0
        feature = node_input.narrow(1, 0, 1)
        feature = self.norm_l0(feature)
        out.append(feature)

        # for L > 0
        if self.lmax > 0:
            num_m_components = (self.lmax + 1) ** 2
            feature = node_input.narrow(1, 1, num_m_components - 1)

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == "norm":
                feature_norm = feature.pow(2).sum(
                    dim=1, keepdim=True
                )  # [N, 1, C]
            elif self.normalization == "component":
                if self.std_balance_degrees:
                    feature_norm = feature.pow(
                        2
                    )  # [N, (L_max + 1)**2 - 1, C], without L = 0
                    feature_norm = torch.einsum(
                        "nic, ia -> nac",
                        feature_norm,
                        self.balance_degree_weight,
                    )  # [N, 1, C]
                else:
                    feature_norm = feature.pow(2).mean(
                        dim=1, keepdim=True
                    )  # [N, 1, C]

            feature_norm = torch.mean(
                feature_norm, dim=2, keepdim=True
            )  # [N, 1, 1]
            feature_norm = (feature_norm + self.eps).pow(-0.5)

            for lval in range(1, self.lmax + 1):
                start_idx = lval**2
                length = 2 * lval + 1
                feature = node_input.narrow(
                    1, start_idx, length
                )  # [N, (2L + 1), C]
                if self.affine:
                    weight = self.affine_weight.narrow(
                        0, (lval - 1), 1
                    )  # [1, C]
                    weight = weight.view(1, 1, -1)  # [1, 1, C]
                    feature_scale = feature_norm * weight  # [N, 1, C]
                else:
                    feature_scale = feature_norm
                feature = feature * feature_scale
                out.append(feature)

        out = torch.cat(out, dim=1)
        return out
    
    
if __name__ == "__main__":
    model = EquivariantLayerNormArraySphericalHarmonics(
        lmax = 2,
        num_channels=128,

    )
    x = torch.randn(20,1+3+5,128)
    model(x)