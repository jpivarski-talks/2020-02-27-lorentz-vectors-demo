import sys

if "/home/jpivarski/irishep/awkward-1.0" not in sys.path:
    sys.path.insert(0, "/home/jpivarski/irishep/awkward-1.0")

import uproot
import awkward1 as ak
import numpy as np
import numba as nb

# Get some data from Uproot.
tree = uproot.open("../uproot/tests/samples/HZZ.root")["events"]
x, y, z, t = tree.arrays(["Muon_Px", "Muon_Py", "Muon_Pz", "Muon_E"], outputtype=tuple)
offsets = ak.layout.Index64(x.offsets)
content = ak.layout.RecordArray({
    "x": ak.layout.NumpyArray(x.content), "y": ak.layout.NumpyArray(y.content), 
    "z": ak.layout.NumpyArray(z.content), "t": ak.layout.NumpyArray(t.content)},
    parameters={"__record__": "LorentzXYZ", "__typestr__": "Lxyz"})

# This array is generic: it doesn't know what records labeled "LorentzXYZ" mean.
example = ak.Array(ak.layout.ListOffsetArray64(offsets, content))
print(repr(example))

# These functions can be reused for LorentzXYZ objects, LorentzXYZArray arrays, and Numba.
def lorentz_xyz_pt(rec):
    return np.sqrt(rec.x**2 + rec.y**2)

def lorentz_xyz_eta(rec):
    return np.arcsinh(rec.z / np.sqrt(rec.x**2 + rec.y**2))

def lorentz_xyz_phi(rec):
    return np.arctan2(rec.y, rec.x)

def lorentz_xyz_mass(rec):
    return np.sqrt(rec.t**2 - rec.x**2 - rec.y**2 - rec.z**2)

# This function only works as a ufunc overload, but it 
def lorentz_add_xyz_xyz(left, right):
    x = ak.layout.NumpyArray(np.asarray(left["x"]) + np.asarray(right["x"]))
    y = ak.layout.NumpyArray(np.asarray(left["y"]) + np.asarray(right["y"]))
    z = ak.layout.NumpyArray(np.asarray(left["z"]) + np.asarray(right["z"]))
    t = ak.layout.NumpyArray(np.asarray(left["t"]) + np.asarray(right["t"]))
    return ak.layout.RecordArray({"x": x, "y": y, "z": z, "t": t},
               parameters={"__record__": "LorentzXYZ", "__typestr__": "Lxyz"})

# Many of the functions can be used for records and arrays of them, so use a base class.
class LorentzXYZCommon:
    @property
    def pt(self):
        with np.errstate(invalid="ignore"):
            return lorentz_xyz_pt(self)

    @property
    def eta(self):
        with np.errstate(invalid="ignore"):
            return lorentz_xyz_eta(self)

    @property
    def phi(self):
        with np.errstate(invalid="ignore"):
            return lorentz_xyz_phi(self)

    @property
    def mass(self):
        with np.errstate(invalid="ignore"):
            return lorentz_xyz_mass(self)

class LorentzXYZ(ak.Record, LorentzXYZCommon):
    def __repr__(self):
        return "Lxyz({0:.3g} {1:.3g} {2:.3g} {3:.3g})".format(self.x, self.y, self.z, self.t)

class LorentzXYZArray(ak.Array, LorentzXYZCommon):
    pass

# Define some behaviors for Lorentz vectors.
lorentzbehavior = dict(ak.behavior)

# Any records with __record__ = "LorentzXYZ" will be mapped to LorentzXYZ instances.
lorentzbehavior["LorentzXYZ"] = LorentzXYZ

# Any arrays containing such records (any number of levels deep) will be LorentsXYZArrays.
lorentzbehavior["*", "LorentzXYZ"] = LorentzXYZArray

# The NumPy ufunc for "add" will use our definition for __record__ = "LorentzXYZ".
lorentzbehavior[np.add, "LorentzXYZ", "LorentzXYZ"] = lorentz_add_xyz_xyz

# This new array understands that data labeled "LorentzXYZ" should have the above methods.
example2 = ak.Array(example, behavior=lorentzbehavior)

print(repr(example2))
print(repr(example2[0, 0]))
print(repr(example2[0, 0].mass))
print(repr(example2.mass))

# We need a "ak.sizes" function with a simpler interface than this...
hastwo = ak.count(example2, axis=-1).x >= 2

print(example2[hastwo, 0] + example2[hastwo, 1])
print((example2[hastwo, 0] + example2[hastwo, 1]).mass)

# Now for Numba:
def lorentz_xyz_pt_typer(viewtype):
    return nb.float64

def lorentz_xyz_pt_lower(context, builder, sig, args):
    return context.compile_internal(builder, lorentz_xyz_pt, sig, args)

lorentzbehavior["__numba_typer__", "LorentzXYZ", "pt"] = lorentz_xyz_pt_typer
lorentzbehavior["__numba_lower__", "LorentzXYZ", "pt"] = lorentz_xyz_pt_lower

example3 = ak.Array(example, behavior=lorentzbehavior)

@nb.njit
def do_it_in_numba(input, output):
    for event in input:
        output.beginlist()

        for muon in event:
            output.beginrecord()
            output.field("muon")
            output.append(muon)
            output.field("pt")
            output.append(muon.pt)
            output.endrecord()

        output.endlist()

output = ak.FillableArray(behavior=lorentzbehavior)
do_it_in_numba(example3, output)

print(output.snapshot())
