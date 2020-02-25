import sys

if "/home/jpivarski/irishep/awkward-1.0" not in sys.path:
    sys.path.insert(0, "/home/jpivarski/irishep/awkward-1.0")

import uproot
import awkward1 as ak
import numpy as np
import numba as nb

tree = uproot.open("../uproot/tests/samples/HZZ.root")["events"]
x, y, z, t = tree.arrays(["Muon_Px", "Muon_Py", "Muon_Pz", "Muon_E"], outputtype=tuple)
offsets = ak.layout.Index64(x.offsets)
content = ak.layout.RecordArray({
    "x": ak.layout.NumpyArray(x.content), "y": ak.layout.NumpyArray(y.content), 
    "z": ak.layout.NumpyArray(z.content), "t": ak.layout.NumpyArray(t.content)})
content.setparameter("__record__", "LorentzXYZ")
content.setparameter("__typestr__", "Lxyz")
example = ak.Array(ak.layout.ListOffsetArray64(offsets, content))

print(repr(example))

def lorentz_xyz_pt(rec):
    return np.sqrt(rec.x**2 + rec.y**2)

def lorentz_xyz_eta(rec):
    return np.arcsinh(rec.z / np.sqrt(rec.x**2 + rec.y**2))

def lorentz_xyz_phi(rec):
    return np.arctan2(rec.y, rec.x)

def lorentz_xyz_mass(rec):
    return np.sqrt(rec.t**2 - rec.x**2 - rec.y**2 - rec.z**2)

def lorentz_add_xyz_xyz(left, right):
    x = ak.layout.NumpyArray(np.asarray(left["x"]) + np.asarray(right["x"]))
    y = ak.layout.NumpyArray(np.asarray(left["y"]) + np.asarray(right["y"]))
    z = ak.layout.NumpyArray(np.asarray(left["z"]) + np.asarray(right["z"]))
    t = ak.layout.NumpyArray(np.asarray(left["t"]) + np.asarray(right["t"]))

    out = ak.layout.RecordArray({"x": x, "y": y, "z": z, "t": t})
    out.setparameter("__record__", "LorentzXYZ")
    out.setparameter("__typestr__", "Lxyz")
    return out

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

lorentzbehavior = dict(ak.behavior)
lorentzbehavior["LorentzXYZ"] = LorentzXYZ
lorentzbehavior["*", "LorentzXYZ"] = LorentzXYZArray
lorentzbehavior[np.add, "LorentzXYZ", "LorentzXYZ"] = lorentz_add_xyz_xyz

example2 = ak.Array(example, behavior=lorentzbehavior)

print(repr(example2))
print(repr(example2[0, 0]))
print(repr(example2[0, 0].mass))
print(repr(example2.mass))
print(example2[0:10, 0] + example2[0:10, 0])
print((example2[0:10, 0] + example2[0:10, 0]).mass)
