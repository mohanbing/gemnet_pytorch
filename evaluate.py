import numpy as np
import os
import yaml
import torch
import ast

from gemnet.training.data_provider import DataProvider
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["AUTOGRAPH_VERBOSITY"] = "1"

# Set up logger
import logging
logger = logging.getLogger()
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s (%(levelname)s): %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel("INFO")

# import tensorflow as tf
# # TensorFlow logging verbosity
# tf.get_logger().setLevel("WARN")
# tf.autograph.set_verbosity(1)

# GemNet imports
from gemnet.model.gemnet import GemNet
from gemnet.training.data_container import DataContainer

class Molecule(DataContainer):
    """
    Implements the DataContainer but for a single molecule. Requires custom init method.
    """
    def __init__(self, R, Z, cutoff, int_cutoff, triplets_only=False):
        self.index_keys = [
            "batch_seg",
            "id_undir",
            "id_swap",
            "id_c",
            "id_a",
            "id3_expand_ba",
            "id3_reduce_ca",
            "Kidx3",
        ]
        if not triplets_only:
            self.index_keys += [
                "id4_int_b",
                "id4_int_a",
                "id4_reduce_ca",
                "id4_expand_db",
                "id4_reduce_cab",
                "id4_expand_abd",
                "Kidx4",
                "id4_reduce_intm_ca",
                "id4_expand_intm_db",
                "id4_reduce_intm_ab",
                "id4_expand_intm_ab",
            ]
        self.triplets_only = triplets_only
        self.cutoff = cutoff
        self.int_cutoff = int_cutoff
        self.keys = ["N", "Z", "R", "F", "E"]

        assert R.shape == (len(Z), 3)
        self.R = R
        self.Z = Z
        self.N = np.array([len(Z)], dtype=np.int32)
        self.E = np.zeros(1, dtype=np.float32).reshape(1, 1)
        self.F = np.zeros((len(Z), 3), dtype=np.float32)

        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])
        self.addID = False
        self.dtypes, dtypes2 = self.get_dtypes()
        self.dtypes.update(dtypes2)  # merge all dtypes in single dict

    def get(self):
        """
        Get the molecule representation in the expected format for the GemNet model.
        """
        data = self.__getitem__(0)
        for var in ["E", "F"]:
            data.pop(var)  # not needed i.e.e not kown -> want to calculate this
        return data

# Model setup
scale_file = "./scaling_factors.json"
pytorch_weights_file = "./pretrained/best/model.pth"
# depends on GemNet model that is loaded
triplets_only = False
direct_forces = False
cutoff = 5.0
int_cutoff = 10.0

# Data setup
from ase.build import molecule as ase_molecule_db

mol = ase_molecule_db('C7NH5')
R   = mol.get_positions()
Z   = mol.get_atomic_numbers()

molecule = Molecule(
    R, Z, cutoff=cutoff, int_cutoff=int_cutoff, triplets_only=triplets_only
)

model = GemNet(
    num_spherical=7,
    num_radial=6,
    num_blocks=4,
    emb_size_atom=128,
    emb_size_edge=128,
    emb_size_trip=64,
    emb_size_quad=32,
    emb_size_rbf=16,
    emb_size_cbf=16,
    emb_size_sbf=32,
    emb_size_bil_trip=64,
    emb_size_bil_quad=32,
    num_before_skip=1,
    num_after_skip=1,
    num_concat=1,
    num_atom=2,
    num_targets=1,
    cutoff=cutoff,
    int_cutoff=int_cutoff,  # no effect for GemNet-(d)T
    scale_file=scale_file,
    triplets_only=triplets_only,
    direct_forces=direct_forces,
)
model.load_weights(pytorch_weights_file)

energy, forces = model.predict(molecule.get())

print("Energy [eV]", energy)
print("Forces [eV/°A]", forces)


with open('config.yaml', 'r') as c:
    config = yaml.safe_load(c)
    
# For strings that yaml doesn't parse (e.g. None)
for key, val in config.items():
    if type(val) is str:
        try:
            config[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass

test_dataset = config["test_dataset"]
test_data_container = DataContainer(
    test_dataset, cutoff=cutoff, int_cutoff=int_cutoff, triplets_only=triplets_only
)
test = {}
batch_size = 32
data_seed = config["data_seed"]
mve = config["mve"]

num_val = len(test_data_container)
logging.info(f"Test data size: {num_val}")
test_data_provider = DataProvider(
    test_data_container,
    0,
    num_val,
    batch_size,
    seed=data_seed,
    shuffle=True,
    random_split=True,
)

test["dataset_iter"] = test_data_provider.get_dataset("test")

def predict(inputs):
    energy, forces = model(inputs)

    if mve:
        mean_energy = energy[:, :1]
        var_energy = torch.nn.functional.softplus(energy[:, 1:])
        mean_forces = forces[:, 0, :]
        var_forces = torch.nn.functional.softplus(forces[:, 1, :])
        return mean_energy, var_energy, mean_forces, var_forces
    else:
        if len(forces.shape) == 3:
            forces = forces[:, 0]
        return energy, None, forces, None

def dict2device(data, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key in data:
        data[key] = data[key].to(device)
    return data

def get_mae(self, targets, pred):
    """
    Mean Absolute Error
    """
    return torch.nn.functional.l1_loss(pred, targets, reduction="mean")

def test_on_batch(dataset_iter):
    model.eval()
    with torch.no_grad():
        inputs, targets = next(dataset_iter)
        # push to GPU if available
        inputs, targets = dict2device(inputs), dict2device(targets)
        energy, _, forces, _ = predict(inputs)
    return (energy, forces), targets

total_mae_energy = 0
total_mae_forces = 0
count = 0
for i in range(int(np.ceil(num_val / batch_size))):
    print("Idx: {}".format(i))
    # predicted_tup, targets =  test_on_batch(test["dataset_iter"])

    inputs, targets = next(test["dataset_iter"])
    inputs, targets = dict2device(inputs), dict2device(targets)
    
    with torch.no_grad():
        energy, forces = model.predict(inputs)

    energy_mae = get_mae(targets["E"], energy)
    forces_mae = get_mae(targets["F"], forces)
    print("E: {}, F: {}, Idx: {}".format(energy_mae, forces_mae, i))

    total_mae_energy += energy_mae
    total_mae_forces += forces_mae
    count +=1

print("Average MAE for Energy: {}".format(total_mae_energy/count))
print("Average MAE for Forces: {}".format(total_mae_forces/count))



