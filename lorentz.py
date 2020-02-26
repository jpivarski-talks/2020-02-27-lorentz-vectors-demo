import sys
import operator
import json

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
    "x": ak.layout.NumpyArray(x.content.astype(np.float64)),
    "y": ak.layout.NumpyArray(y.content.astype(np.float64)), 
    "z": ak.layout.NumpyArray(z.content.astype(np.float64)),
    "t": ak.layout.NumpyArray(t.content.astype(np.float64))},
    parameters={"__record__": "LorentzXYZ", "__typestr__": "Lxyz"})

# I like "LorentzXYZ" as a name for Cartesian Lorentz vectors. It can recognizably
# be shortened to "Lxyz" and it invites the cylindrical form to be "LorentzCyl/Lcyl".
#
# They should be interchangeable: having the same methods/properties and freely
# returning whichever form is most convenient. Source vectors would likely be Lcyl
# and adding them would likely return Lxyz, for instance.

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
def typer_lorentz_xyz_pt(viewtype):
    return nb.float64

def lower_lorentz_xyz_pt(context, builder, sig, args):
    return context.compile_internal(builder, lorentz_xyz_pt, sig, args)

lorentzbehavior["__numba_typer__", "LorentzXYZ", "pt"] = typer_lorentz_xyz_pt
lorentzbehavior["__numba_lower__", "LorentzXYZ", "pt"] = lower_lorentz_xyz_pt

# If we wanted a method (with arguments determined in the typer), the signature would be:
# 
#     lorentzbehavior["__numba_lower__", "LorentzXYZ", "pt", ()] = ...

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

# We can define binary operations (operator.add being the one we want most)...
def typer_lorentz_xyz_eq(binop, left, right):
    return nb.boolean(left, right)

def lower_lorentz_xyz_eq(context, builder, sig, args):
    def compute(left, right):
        return abs(left.x - right.x) + abs(left.y - right.y) + abs(left.z - right.z) + abs(left.t - right.t) < 0.001
    return context.compile_internal(builder, compute, sig, args)

lorentzbehavior["__numba_typer__", "LorentzXYZ", operator.eq, "LorentzXYZ"] = typer_lorentz_xyz_eq
lorentzbehavior["__numba_lower__", "LorentzXYZ", operator.eq, "LorentzXYZ"] = lower_lorentz_xyz_eq

example4 = ak.Array(example, behavior=lorentzbehavior)

@nb.njit
def check_equality(input, output):
    for muons in input:
        output.beginlist()

        for i in range(len(muons)):
            output.beginlist()
            for j in range(i, len(muons)):
                output.append(muons[i] == muons[j])
            output.endlist()

        output.endlist()

output = ak.FillableArray(behavior=lorentzbehavior)
check_equality(example4, output)

print(output.snapshot())

# The trouble with operator.add is that it returns new objects.
# 
# The records we have been dealing with are not free-floating objects; they're views
# into the arrays, and Awkward Arrays can't be created in Numba (that restriction gives
# us a lot of freedom and this model favors the development of a functional language).
# 
# So we need to create a new Numba type that is a free-floating LorentzXYZCommon.
# Fortunately, that's s more conventional task and serves as a good introduction to Numba.

class LorentzXYZFree(LorentzXYZCommon):
    def __init__(self, x, y, z, t):
        self.x = x
        self.y = y
        self.z = z
        self.t = t

    def __repr__(self):
        return "Lxyz({0:.3g} {1:.3g} {2:.3g} {3:.3g})".format(self.x, self.y, self.z, self.t)

    def __getitem__(self, attr):
        # It has to behave the same way as the bound objects or users will get confused.
        if attr in ("x", "y", "z", "t"):
            return getattr(self, attr)
        else:
            raise ValueError("key {0} does not exist (not in record)".format(json.dumps(attr)))

@nb.extending.typeof_impl.register(LorentzXYZFree)
def typeof_LorentzXYZFree(obj, c):
    return LorentzXYZType()

class LorentzXYZType(nb.types.Type):
    def __init__(self):
        # Type names have to be unique identifiers; they determine whether Numba
        # will recompile a function with new types.
        super(LorentzXYZType, self).__init__(name="LorentzXYZType()")

@nb.extending.register_model(LorentzXYZType)
class LorentzXYZModel(nb.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        # This is the C-style struct that will be used wherever LorentzXYZ are needed.
        members = [("x", nb.float64),
                   ("y", nb.float64),
                   ("z", nb.float64),
                   ("t", nb.float64)]
        super(LorentzXYZModel, self).__init__(dmm, fe_type, members)

@nb.extending.unbox(LorentzXYZType)
def unbox_LorentzXYZ(lxyztype, lxyzobj, c):
    # How to turn LorentzXYZFree Python objects into LorentzXYZModel structs.
    x_obj = c.pyapi.object_getattr_string(lxyzobj, "x")
    y_obj = c.pyapi.object_getattr_string(lxyzobj, "y")
    z_obj = c.pyapi.object_getattr_string(lxyzobj, "z")
    t_obj = c.pyapi.object_getattr_string(lxyzobj, "t")

    # "values" are raw LLVM code; "proxies" have getattr/setattr logic to access fields.
    outproxy = c.context.make_helper(c.builder, lxyztype)

    # https://github.com/numba/numba/blob/master/numba/core/pythonapi.py
    outproxy.x = c.pyapi.float_as_double(x_obj)
    outproxy.y = c.pyapi.float_as_double(y_obj)
    outproxy.z = c.pyapi.float_as_double(z_obj)
    outproxy.t = c.pyapi.float_as_double(t_obj)

    # Yes, we're in that world...
    c.pyapi.decref(x_obj)
    c.pyapi.decref(y_obj)
    c.pyapi.decref(z_obj)
    c.pyapi.decref(t_obj)

    is_error = nb.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return nb.extending.NativeValue(outproxy._getvalue(), is_error)

@nb.extending.box(LorentzXYZType)
def box_LorentzXYZ(lxyztype, lxyzval, c):
    # This proxy is initialized with a value, used for getattr, rather than setattr.
    inproxy = c.context.make_helper(c.builder, lxyztype, lxyzval)
    x_obj = c.pyapi.float_from_double(inproxy.x)
    y_obj = c.pyapi.float_from_double(inproxy.y)
    z_obj = c.pyapi.float_from_double(inproxy.z)
    t_obj = c.pyapi.float_from_double(inproxy.t)

    # The way we get Python objects into this lowered world is by pickling them.
    LorentzXYZFree_obj = c.pyapi.unserialize(c.pyapi.serialize_object(LorentzXYZFree))

    out = c.pyapi.call_function_objargs(LorentzXYZFree_obj, (x_obj, y_obj, z_obj, t_obj))

    c.pyapi.decref(LorentzXYZFree_obj)
    c.pyapi.decref(x_obj)
    c.pyapi.decref(y_obj)
    c.pyapi.decref(z_obj)
    c.pyapi.decref(t_obj)

    return out

# Now we've defined enough that our objects can go into and come out of Numba.

testit = LorentzXYZFree(1, 2, 3, 4)
print(testit)

@nb.njit
def pass_through(obj):
    return obj

print(testit, pass_through(testit))

# Notice that the original has int fields and the passed-through has floats:
print(testit.x, pass_through(testit).x)

# Defining an in-Numba constructor is a separate thing.
@nb.extending.type_callable(LorentzXYZFree)
def typer_LorentzXYZFree_constructor(context):
    def typer(x, y, z, t):
        if x == nb.types.float64 and y == nb.types.float64 and z == nb.types.float64 and t == nb.types.float64:
            return LorentzXYZType()
    return typer

@nb.extending.lower_builtin(LorentzXYZFree, nb.types.float64, nb.types.float64, nb.types.float64, nb.types.float64)
def lower_LorentzXYZFree_constructor(context, builder, sig, args):
    rettype, (xtype, ytype, ztype, ttype) = sig.return_type, sig.args
    xval, yval, zval, tval = args

    outproxy = context.make_helper(builder, rettype)
    outproxy.x = xval
    outproxy.y = yval
    outproxy.z = zval
    outproxy.t = tval

    return outproxy._getvalue()

# Test it!

@nb.njit
def test_constructor():
    return LorentzXYZFree(1.1, 2.2, 3.3, 4.4)

print(test_constructor())
        
# Now it's time to define the methods and properties.

# To simply map model attributes to user-accessible properties, use a macro.
nb.extending.make_attribute_wrapper(LorentzXYZType, "x", "x")
nb.extending.make_attribute_wrapper(LorentzXYZType, "y", "y")
nb.extending.make_attribute_wrapper(LorentzXYZType, "z", "z")
nb.extending.make_attribute_wrapper(LorentzXYZType, "t", "t")

# For more general cases, there's an AttributeTemplate.
@nb.typing.templates.infer_getattr
class typer_LorentzXYZ_methods(nb.typing.templates.AttributeTemplate):
    key = LorentzXYZType

    def generic_resolve(self, lxyztype, attr):
        if attr == "pt":
            return nb.float64
        elif attr == "eta":
            return nb.float64
        elif attr == "phi":
            return nb.float64
        elif attr == "mass":
            return nb.float64
        else:
            # typers that return None defer to other typers.
            return None

    # If we had any methods with arguments, this is how we'd do it.
    # 
    # @nb.typing.templates.bound_function("pt")
    # def pt_resolve(self, lxyztype, args, kwargs):
    #     ...

# To lower these functions, we can duck-type the Python functions above.
# Since they're defined in terms of NumPy functions, they apply to
#
#    * Python scalars
#    * NumPy arrays
#    * Awkward arrays
#    * lowered Numba values

@nb.extending.lower_getattr(LorentzXYZType, "pt")
def lower_LorentzXYZ_pt(context, builder, lxyztype, lxyzval):
    return context.compile_internal(builder, lorentz_xyz_pt, nb.float64(lxyztype), (lxyzval,))

@nb.extending.lower_getattr(LorentzXYZType, "eta")
def lower_LorentzXYZ_eta(context, builder, lxyztype, lxyzval):
    return context.compile_internal(builder, lorentz_xyz_eta, nb.float64(lxyztype), (lxyzval,))

@nb.extending.lower_getattr(LorentzXYZType, "phi")
def lower_LorentzXYZ_phi(context, builder, lxyztype, lxyzval):
    return context.compile_internal(builder, lorentz_xyz_phi, nb.float64(lxyztype), (lxyzval,))

@nb.extending.lower_getattr(LorentzXYZType, "mass")
def lower_LorentzXYZ_mass(context, builder, lxyztype, lxyzval):
    return context.compile_internal(builder, lorentz_xyz_mass, nb.float64(lxyztype), (lxyzval,))

# And the __getitem__ access...
@nb.typing.templates.infer_global(operator.getitem)
class typer_LorentzXYZ_getitem(nb.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 2 and len(kwargs) == 0 and isinstance(args[0], LorentzXYZType):
            # Only accept compile-time constants. It's a fair restriction.
            if isinstance(args[1], nb.types.StringLiteral):
                if args[1].literal_value in ("x", "y", "z", "t"):
                    return nb.float64(*args)

@nb.extending.lower_builtin(operator.getitem, LorentzXYZType, nb.types.StringLiteral)
def lower_getitem_LorentzXYZ(context, builder, sig, args):
    rettype, (lxyztype, wheretype) = sig.return_type, sig.args
    lxyzval, whereval = args

    inproxy = context.make_helper(builder, lxyztype, lxyzval)

    # The value of a StringLiteral is in its compile-time type.
    if wheretype.literal_value == "x":
        return inproxy.x
    elif wheretype.literal_value == "y":
        return inproxy.y
    elif wheretype.literal_value == "z":
        return inproxy.z
    elif wheretype.literal_value == "t":
        return inproxy.t

# Now we can use all of these. LorentzXYZFree is as fully functional as LorentzXYZ.

@nb.njit
def try_it_out(testit):
    return testit.x, testit["x"], testit.pt, testit.eta, testit.phi, testit.mass

print(try_it_out(testit))

# Finally, we want to be able to append LorentzXYZFree to a FillableArray, as though
# it were an attached LorentzXYZ. This doesn't need a typer; the types are obvious.

def lower_FillableArray_append_LorentzXYZ(context, builder, sig, args):
    def doit(output, lxyz):
        output.beginrecord("LorentzXYZ")
        output.field("x")
        output.real(lxyz.x)
        output.field("y")
        output.real(lxyz.y)
        output.field("z")
        output.real(lxyz.z)
        output.field("t")
        output.real(lxyz.t)
        output.endrecord()
    return context.compile_internal(builder, doit, sig, args)

lorentzbehavior["__numba_lower__", ak.FillableArray.append, LorentzXYZType] = lower_FillableArray_append_LorentzXYZ

# Attaching free objects to a FillableArray doesn't look any different to the user.

@nb.njit
def fill_it(testit, output):
    output.append(testit)
    output.append(testit)
    
output = ak.FillableArray(behavior=lorentzbehavior)
fill_it(testit, output)

print(output.snapshot())

# Now that we have free Lorentz vectors, we can finally define addition.

def typer_lorentz_xyz_add(binop, left, right):
    return LorentzXYZType()(left, right)

def lower_lorentz_xyz_add(context, builder, sig, args):
    def compute(left, right):
        return LorentzXYZFree(left.x + right.x, left.y + right.y, left.z + right.z, left.t + right.t)
    return context.compile_internal(builder, compute, sig, args)

lorentzbehavior["__numba_typer__", "LorentzXYZ", operator.add, "LorentzXYZ"] = typer_lorentz_xyz_add
lorentzbehavior["__numba_lower__", "LorentzXYZ", operator.add, "LorentzXYZ"] = lower_lorentz_xyz_add

@nb.njit
def test_add(input):
    for muons in input:
        for i in range(len(muons)):
            for j in range(i + 1, len(muons)):
                return muons[i] + muons[j]

example5 = ak.Array(example, behavior=lorentzbehavior)

print(test_add(example5))
