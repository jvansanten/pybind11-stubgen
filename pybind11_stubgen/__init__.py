import ast
import builtins
import copy
import dataclasses
import importlib
import inspect
import itertools
import logging
import os
import re
import sys
from types import ModuleType
import warnings
from argparse import ArgumentParser
import typing
from typing import Any, Callable, ClassVar, Dict, List, Optional, Set, Type, Union
from collections.abc import Sequence, Mapping

logger = logging.getLogger(__name__)

_visited_objects = []

# A list of function docstring pre-processing hooks
function_docstring_preprocessing_hooks: List[Callable[[str], str]] = []

# A list of signature post-processing hooks
function_signature_postprocessing_hooks: list[Callable[["FunctionSignature"], None]] = []

function_overload_filters: list[Callable[[list["FunctionSignature"]], list["FunctionSignature"]]] = []

def flip_arg_annotations(doc: str) -> str:
    """
    replace "(type)arg" with "arg: type". type may contain qualifiers, e.g. "foo.bar.type"
    """
    return re.sub(
        "\(((?:[A-Za-z_]\w*\.)*(?:[A-Za-z_]\w*)+)\)([A-Za-z_]\w*)",
        r"\2: \1",
        doc
    )

def strip_trailing_colon(doc: str) -> str:
    """
    Strip trailing colon from boost-python-generated signature
    """
    pattern = "(-> [A-Za-z_]\w*).*$"
    repl = r"\1"
    return re.sub(pattern, repl, doc, flags=re.MULTILINE)    

OBJECT_PROTOCOL_RETURN_TYPES = {
    "__init__": "None",
    "__ge__": "bool",
    "__gt__": "bool",
    "__lt__": "bool",
    "__le__": "bool",
    "__eq__": "bool",
    "__ne__": "bool",
    "__repr__": "str",
    "__str__": "str",
}

from .boost_python_signature import transform_signatures
function_docstring_preprocessing_hooks.append(transform_signatures)

from functools import partial, reduce, cache

# strip trailing colon from boost-python-generated signature 
function_docstring_preprocessing_hooks.append(partial(re.compile("(-> (?:[A-Za-z_]\w*\.)*(?:[A-Za-z_]\w*)+):\s*?$", flags=re.MULTILINE).sub, r"\1"))

def remove_shadowing_overloads(signatures: list["FunctionSignature"]) -> list["FunctionSignature"]:
    """
    Remove int overloads if a float overload is present. mypy assumes that float
    captures int, even though the conversion loses precision

    Furthermore, pure_virtual adds a default implementation that claims to
    return None, but actually just raises an exception. Remove it.
    """
    if len(signatures) > 1:
        argtypes = [set(s.argtypes.items()) for s in signatures]
        common = reduce(set.intersection, argtypes[1:], argtypes[0])
        unique = [tuple(dict(a.difference(common)).values()) for a in argtypes]
        skip = set()
        if ("float",) in unique and ("int",) in unique:
            skip.add(unique.index(("int",)))
        else:
            groups = {}
            for i, uargs in enumerate(unique):
                if not uargs in groups:
                    groups[uargs] = [i]
                else:
                    groups[uargs].append(i)
            for indices in groups.values():
                if len(indices) > 1:
                    for i in reversed(sorted(indices)):
                        if signatures[i].rtype == "None":
                            skip.add(i)
                            break
        if skip:
            return [s for i,s in enumerate(signatures) if i not in skip]

    return signatures

def qualify_default_values(sig: "FunctionSignature") -> None:
    """
    Replace default args of the form
    
    icecube._dataclasses.I3Position=I3Position(0,0,0)

    with

    icecube._dataclasses.I3Position=icecube._dataclasses.I3Position(0,0,0)
    """
    for arg in sig._args:
        if arg.default is None:
            continue
        default_text = ast.unparse(arg.default)
        if "." in arg.annotation and not default_text.startswith(arg.annotation):
            klass = arg.get_class(sig.module_name)
            qualified = re.sub(f"^({re.escape(klass.__name__)})", f"{klass.__module__}.{klass.__qualname__}", default_text)
            if qualified == default_text:
                qualified = re.sub("^(icetray.I3Frame)(.*)", r"icecube._icetray.I3Frame\2", qualified)
            arg.default = ast.parse(qualified)

def _type_or_union(klass: Union[Type, tuple[Type, ...]]):
    if klass is None:
        return Any
    elif isinstance(klass, Type):
        return get_container_equivalent(klass) or klass
    elif isinstance(klass, tuple):
        return Union[klass]
    else:
        return klass

def get_container_equivalent(klass: Type):
    """Replace container an annotation that covers the types implicitly convertible to that container"""
    if klass in _container_equivalents:
        return _container_equivalents[klass]
    if hasattr(klass, "__key_type__"):
        # std::map
        return Mapping[_type_or_union(klass.__key_type__()), _type_or_union(klass.__value_type__())]
    if hasattr(klass, "__value_type__") and not (hasattr(klass, "pre_order_iterator") or hasattr(klass, "pre_order_iter")):
        # std::vector, but none of the various trees
        return Sequence[_type_or_union(klass.__value_type__())]

def strip_current_module_name(obj, module_name):
    regex = r"{}\.(\w+)".format(module_name.replace(".", r"\."))
    return re.sub(regex, r"\g<1>", obj)

def type_args(obj, module_name):
    args = typing.get_args(obj)
    if args:
        return f'[{",".join(strip_current_module_name(StubsGenerator.fully_qualified_name(arg), module_name) for arg in args)}]'
    return ''

def fully_qualified_type_string(obj, current_module_name):
    return (
        strip_current_module_name(
            StubsGenerator.fully_qualified_name(
                Union[obj] if isinstance(obj, tuple) else obj
            ),
        current_module_name)
    )

def replace_container_types(sig: "FunctionSignature") -> None:
    for arg in sig._args[1:]:
        if arg.annotation is None:
            continue
        if equivalent := get_container_equivalent(arg.get_class(sig.module_name)):
            arg.annotation = repr(equivalent)

def replace_object_protocol_rtypes(sig: "FunctionSignature") -> None:
    sig.rtype = OBJECT_PROTOCOL_RETURN_TYPES.get(sig.name, sig.rtype)

def treat_object_as_any(sig: "FunctionSignature") -> None:
    for arg in sig._args:
        if arg.annotation == "object":
            arg.annotation = "typing.Any"
    if sig.rtype == "object":
        sig.rtype = "typing.Any"

function_overload_filters.append(remove_shadowing_overloads)

function_signature_postprocessing_hooks.append(replace_object_protocol_rtypes)
function_signature_postprocessing_hooks.append(qualify_default_values)
function_signature_postprocessing_hooks.append(replace_container_types)
function_signature_postprocessing_hooks.append(treat_object_as_any)

def _find_str_end(s, start):
    for i in range(start + 1, len(s)):
        c = s[i]
        if c == "\\":  # skip escaped chars
            continue
        if c == s[start]:
            return i
    return -1


def _is_balanced(s):
    closing = {"(": ")", "{": "}", "[": "]"}

    stack = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in "\"'":
            # TODO: handle triple-quoted strings too
            i = _find_str_end(s, i)
            if i < 0:
                return False
        if c in closing:
            stack.append(closing[c])
        elif stack and stack[-1] == c:
            stack.pop()
        i += 1

    return len(stack) == 0


class DirectoryWalkerGuard(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self.origin = os.getcwd()

    def __enter__(self):
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)

        assert os.path.isdir(self.dirname)
        os.chdir(self.dirname)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.origin)


_default_pybind11_repr_re = re.compile(
    r"(<(?P<class>\w+(\.\w+)*) object at 0x[0-9a-fA-F]+>)|"
    r"(<(?P<enum>\w+(.\w+)*): -?\d+>)"
)


def replace_default_pybind11_repr(line):
    default_reprs = []

    def replacement(m):
        if m.group("class"):
            default_reprs.append(m.group(0))
            return "..."
        return m.group("enum")

    return default_reprs, _default_pybind11_repr_re.sub(replacement, line)


def get_args(function_def_str: str):
    parsed = ast.parse(function_def_str)
    f: ast.FunctionDef = parsed.body[0]
    f.args.defaults
    assert not f.args.kwonlyargs
    assert not f.args.posonlyargs
    defaults = [None] * (len(f.args.args)-len(f.args.defaults)) + f.args.defaults
    for arg, default in zip(f.args.args, defaults):
        annotation = None if arg.annotation is None else ast.unparse(arg.annotation)
        yield Argument(arg.arg, annotation, default)

@dataclasses.dataclass
class Argument:
    name: str
    annotation: str | None
    default: ast.Expr | None

    def __str__(self) -> str:
        out = f"{self.name}"
        if self.annotation is not None:
            out += f": {self.annotation}"
        if self.default is not None:
            out += f"={ast.unparse(self.default)}"
        return out

    @staticmethod
    @cache
    def _get_annotation_class(current_module: ModuleType, annotation: str) -> Type:
        parts = annotation.split(".")
        # LEGB rule
        # name in current module (enclosing scope)
        try:
            ns = current_module
            for k in parts:
                ns = getattr(ns, k)
            return ns
        except AttributeError:
            ...
        # fully-qualified name from some other module
        module_path = []
        for i, k in enumerate(parts):
            module_path.append(k)
            try:
                ns = importlib.import_module(".".join(parts[:i+1]))
                for j in range(i+1, len(parts)):
                    ns = getattr(ns, parts[j])
                return ns
            except ModuleNotFoundError:
                continue
        
        # built-in
        return getattr(builtins, parts[-1])
        

    def get_class(self, current_module: str) -> Optional[Type]:
        # skip subscripted generics, these are never going to be wrapped classes
        if "[" in self.annotation:
            return None
        return self._get_annotation_class(importlib.import_module(current_module), self.annotation)
        

class FunctionSignature(object):
    # When True don't raise an error when invalid signatures/defaultargs are
    # encountered (yes, global variables, blame me)
    ignore_invalid_signature = False
    ignore_invalid_defaultarg = False

    signature_downgrade = True

    # Number of invalid default values found so far
    n_invalid_default_values = 0

    # Number of invalid signatures found so far
    n_invalid_signatures = 0

    @classmethod
    def n_fatal_errors(cls):
        return (
            0 if cls.ignore_invalid_defaultarg else cls.n_invalid_default_values
        ) + (0 if cls.ignore_invalid_signature else cls.n_invalid_signatures)

    def __init__(self, name, module_name, args="*args, **kwargs", rtype="None", validate=True):
        self.name = name
        self.module_name = module_name
        self.args = args
        self.rtype = rtype
        self._args: list[Argument] = []
        self.argtypes: dict[str, str] = {}

        if validate:
            invalid_defaults, self.args = replace_default_pybind11_repr(self.args)
            if invalid_defaults:
                FunctionSignature.n_invalid_default_values += 1
                lvl = (
                    logging.WARNING
                    if FunctionSignature.ignore_invalid_defaultarg
                    else logging.ERROR
                )
                logger.log(
                    lvl, "Default argument value(s) replaced with ellipses (...):"
                )
                for invalid_default in invalid_defaults:
                    logger.log(lvl, "    {}".format(invalid_default))

            function_def_str = "def {sig.name}({sig.args}) -> {sig.rtype}: ...".format(
                sig=self
            )
            try:
                for arg in get_args(function_def_str):
                    self.argtypes[arg.name] = arg.annotation
                    self._args.append(arg)
            except SyntaxError as e:
                FunctionSignature.n_invalid_signatures += 1
                if FunctionSignature.signature_downgrade:
                    self.name = name
                    self.args = "*args, **kwargs"
                    self.rtype = "typing.Any"
                    lvl = (
                        logging.WARNING
                        if FunctionSignature.ignore_invalid_signature
                        else logging.ERROR
                    )
                    logger.log(
                        lvl,
                        "Generated stubs signature is degraded to `(*args, **kwargs) -> typing.Any` for",
                    )
                else:
                    lvl = logging.WARNING
                    logger.warning("Ignoring invalid signature:")
                logger.log(lvl, function_def_str)
                logger.log(lvl, " " * (e.offset - 1) + "^-- Invalid syntax")

    def __eq__(self, other):
        return isinstance(other, FunctionSignature) and (
            self.name,
            self.args,
            self.rtype,
        ) == (other.name, other.args, other.rtype)

    def __hash__(self):
        return hash((self.name, self.args, self.rtype))

    def __repr__(self):
        return f"FunctionSignature({self.name}, args={self.args}, rtype={self.rtype})"

    def __str__(self):
        return f'{self.name}({", ".join(str(arg) for arg in self._args)}) -> {self.rtype}:'

    def split_arguments(self):
        if len(self.args.strip()) == 0:
            return []

        prev_stop = 0
        brackets = 0
        splitted_args = []

        for i, c in enumerate(self.args):
            if c == "[":
                brackets += 1
            elif c == "]":
                brackets -= 1
                assert brackets >= 0
            elif c == "," and brackets == 0:
                splitted_args.append(self.args[prev_stop:i])
                prev_stop = i + 1

        splitted_args.append(self.args[prev_stop:])
        assert brackets == 0
        return splitted_args

    @staticmethod
    def argument_type(arg):
        return arg.split(":")[-1].strip()

    def get_all_involved_types(self):
        types = []
        for t in [self.rtype, *(arg.annotation for arg in self._args if arg.annotation)]:
            types.extend(
                [
                    m[0]
                    for m in re.findall(
                        r"([a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*)",
                        t,
                    )
                ]
            )
        return types


class PropertySignature(object):
    NONE = 0
    READ_ONLY = 1
    WRITE_ONLY = 2
    READ_WRITE = READ_ONLY | WRITE_ONLY

    def __init__(self, rtype, setter_args, access_type):
        self.rtype = rtype
        self.setter_args = setter_args
        self.access_type = access_type

    @property
    def setter_arg_type(self):
        return FunctionSignature.argument_type(
            FunctionSignature("name", self.setter_args).split_arguments()[1]
        )
    
    def __repr__(self):
        return f"PropertySignature(rtype={self.rtype}, setter_args={self.setter_args}, access_type={self.access_type}"

    def get_all_involved_types(self):
        return [self.setter_arg_type, self.rtype]

# If true numpy.ndarray[int32[3,3]] will be reduced to numpy.ndarray
BARE_NUPMY_NDARRAY = False


def replace_numpy_array(match_obj):
    if BARE_NUPMY_NDARRAY:
        return "numpy.ndarray"
    numpy_type = match_obj.group("type")
    # pybind always append size of data type
    if numpy_type in [
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
        "complex32",
        "complex64",
        "longcomplex",
    ]:
        numpy_type = "numpy." + numpy_type

    shape = match_obj.group("shape")
    if shape:
        shape = ", _Shape[{}]".format(shape)
    else:
        shape = ""
    result = r"numpy.ndarray[{type}{shape}]".format(type=numpy_type, shape=shape)
    return result


def replace_typing_types(match):
    # pybind used to have iterator/iterable in place of Iterator/Iterable
    name = match.group("type")
    capitalized = name[0].capitalize() + name[1:]
    return "typing." + capitalized

def preprocess_docstring(doc):
    if doc is None:
        return ""
    for hook in function_docstring_preprocessing_hooks:
        doc = hook(doc)
    return doc

class StubsGenerator(object):
    INDENT = " " * 4

    GLOBAL_CLASSNAME_REPLACEMENTS = {
        re.compile(
            r"numpy.ndarray\[(?P<type>[^\[\]]+)(\[(?P<shape>[^\[\]]+)\])?(?P<extra>[^][]*)\]"
        ): replace_numpy_array,
        re.compile(
            r"(?<!\w)(?P<type>Callable|Dict|[Ii]terator|[Ii]terable|List"
            r"|Optional|Set|Tuple|Union|ItemsView|KeysView|ValuesView)(?!\w)"
        ): replace_typing_types,
        re.compile(r"(?<!\w)(object)(?!\w)"): "typing.Any",
    }

    def parse(self):
        raise NotImplementedError

    def to_lines(self):  # type: () -> List[str]
        raise NotImplementedError

    @staticmethod
    def _indent(line):  # type: (str) -> str
        return StubsGenerator.INDENT + line

    @staticmethod
    def indent(lines):  # type: (str) -> str
        lines = lines.split("\n")
        lines = [StubsGenerator._indent(l) if l else l for l in lines]
        return "\n".join(lines)

    @staticmethod
    def module_from_qualname(qualname):  # type: (str) -> Optional[str]
        """Extract a module name from a fully-qualified name"""
        module_parts = qualname.split(".")
        name_parts = []
        while len(module_parts) > 1:
            name_parts.insert(0, module_parts.pop())
            try:
                module_name = ".".join(module_parts)
                root = importlib.import_module(module_name)
                try:
                    for k in name_parts:
                        root = getattr(root, k)
                    return module_name
                except AttributeError:
                    continue
            except ModuleNotFoundError:
                continue

        return None

    @staticmethod
    def fully_qualified_name(klass):
        module_name = klass.__module__ if hasattr(klass, "__module__") else None
        class_name = getattr(klass, "__qualname__", klass.__name__)

        if module_name == "builtins":
            return class_name
        elif typing.get_args(klass):
            # include type args (e.g. Sequence[int])
            return repr(klass)
        else:
            return "{module}.{klass}".format(module=module_name, klass=class_name)

    @staticmethod
    def apply_classname_replacements(s):  # type: (str) -> Any
        for k, v in StubsGenerator.GLOBAL_CLASSNAME_REPLACEMENTS.items():
            s = k.sub(v, s)
        return s

    @staticmethod
    def function_signatures_from_docstring(
        name, func, module_name
    ):  # type: (str, Any, str) -> List[FunctionSignature]
        try:
            signature_regex = (
                r"(\s*(?P<overload_number>\d+).)"
                r"?\s*{name}\s*\((?P<args>{balanced_parentheses})\)"
                r"\s*->\s*"
                r"(?P<rtype>[^\(\)]+)\s*".format(name=name, balanced_parentheses=".*")
            )
            docstring = preprocess_docstring(func.__doc__)

            signatures = []
            for line in docstring.split("\n"):
                m = re.match(signature_regex, line)
                if m:
                    args = m.group("args")
                    rtype = m.group("rtype")
                    if _is_balanced(args):
                        signatures.append(FunctionSignature(name, module_name, args, rtype))

            # strip module name if provided
            if module_name:
                for sig in signatures:
                    regex = r"{}\.(\w+)".format(module_name.replace(".", r"\."))
                    sig.args = re.sub(regex, r"\g<1>", sig.args)
                    sig.rtype = re.sub(regex, r"\g<1>", sig.rtype)

            for sig in signatures:
                sig.args = StubsGenerator.apply_classname_replacements(sig.args)
                sig.rtype = StubsGenerator.apply_classname_replacements(sig.rtype)
            
            signatures = list(set(signatures))

            for sig in signatures:
                for sighook in function_signature_postprocessing_hooks:
                    sighook(sig)

            for sigfilter in function_overload_filters:
                signatures = sigfilter(signatures)

            return sorted(set(signatures), key=lambda fs: fs.args)
        except AttributeError:
            raise
            return []

    @staticmethod
    def property_signature_from_docstring(
        prop, module_name
    ):  # type:  (Any, str)-> PropertySignature

        getter_rtype = "None"
        setter_args = "None"
        access_type = PropertySignature.NONE

        strip_module_name = False

        if hasattr(prop, "fget") and prop.fget is not None:
            access_type |= PropertySignature.READ_ONLY
            if hasattr(prop.fget, "__doc__") and prop.fget.__doc__ is not None:
                for line in preprocess_docstring(prop.fget.__doc__).split("\n"):
                    if strip_module_name:
                        line = line.replace(module_name + ".", "")
                    m = re.match(
                        r"(\s*(?P<overload_number>\d+)\.)?\s*(\w*)\((?P<args>[^()]*)\)\s*->\s*(?P<rtype>[^()]+)\s*",
                        line,
                    )
                    if m:
                        getter_rtype = m.group("rtype")
                        break

        if hasattr(prop, "fset") and prop.fset is not None:
            access_type |= PropertySignature.WRITE_ONLY
            if hasattr(prop.fset, "__doc__") and prop.fset.__doc__ is not None:
                for line in preprocess_docstring(prop.fset.__doc__).split("\n"):
                    if strip_module_name:
                        line = line.replace(module_name + ".", "")
                    m = re.match(
                        r"(\s*(?P<overload_number>\d+)\.)?\s*(\w*)\((?P<args>[^()]*)\)\s*->\s*(?P<rtype>[^()]+)\s*",
                        line,
                    )
                    if m:
                        args = m.group("args")
                        # replace first argument with self
                        setter_args = ",".join(["self"] + args.split(",")[1:])
                        break
        getter_rtype = StubsGenerator.apply_classname_replacements(getter_rtype)
        setter_args = StubsGenerator.apply_classname_replacements(setter_args)
        return PropertySignature(getter_rtype, setter_args, access_type)

    @staticmethod
    def remove_signatures(docstring):  # type: (str) ->str

        if docstring is None:
            return ""

        docstring = preprocess_docstring(docstring)

        signature_regex = (
            r"(\s*(?P<overload_number>\d+).\s*)"
            r"?{name}\s*\((?P<args>.*)\)\s*(->\s*(?P<rtype>[^\(\)]+)\s*)?".format(
                name=r"\w+"
            )
        )

        lines = docstring.split("\n\n")
        lines = filter(lambda line: line != "Overloaded function.", lines)

        return "\n\n".join(
            filter(lambda line: not re.match(signature_regex, line), lines)
        )

    @staticmethod
    def sanitize_docstring(docstring):  # type: (str) ->str
        docstring = StubsGenerator.remove_signatures(docstring)
        docstring = docstring.rstrip("\n")

        if docstring and re.match(r"^\s*$", docstring):
            docstring = ""

        return docstring

    @staticmethod
    def format_docstring(docstring):
        docstring = inspect.cleandoc("\n" + docstring)
        return StubsGenerator.indent('"""\n{}\n"""'.format(docstring.strip("\n")))

def is_boost_python_enum(klass):
    return any(
        "{module}.{name}".format(module=b.__module__, name=b.__name__) == "Boost.Python.enum"
        for b in klass.__bases__
    )

class AttributeStubsGenerator(StubsGenerator):
    def __init__(self, name, attribute):  # type: (str, Any)-> None
        self.name = name
        self.attr = attribute

    def parse(self):
        if self in _visited_objects:
            return
        _visited_objects.append(self)

    def is_safe_to_use_repr(self, value):
        if value is None or isinstance(value, (int, str)):
            return True
        if isinstance(value, (float, complex)):
            try:
                eval(repr(value))
                return True
            except (SyntaxError, NameError):
                return False
        if isinstance(value, (list, tuple, set)):
            for x in value:
                if not self.is_safe_to_use_repr(x):
                    return False
            return True
        if isinstance(value, dict):
            for k, v in value.items():
                if not self.is_safe_to_use_repr(k) or not self.is_safe_to_use_repr(v):
                    return False
            return True
        return False

    def to_lines(self):  # type: () -> List[str]
        # special case for boost-python enums
        attr_type = type(self.attr)
        if is_boost_python_enum(attr_type):
            return ["{name} = {klass}({repr})".format(name=self.name, klass=self.fully_qualified_name(attr_type), repr=repr(int(self.attr)))]

        if self.is_safe_to_use_repr(self.attr):
            return ["{name} = {repr}".format(name=self.name, repr=repr(self.attr))]

        # special case for modules
        # https://github.com/sizmailov/pybind11-stubgen/issues/43
        if type(self.attr) is type(os) and hasattr(self.attr, "__name__"):
            return ["{name} = {repr}".format(name=self.name, repr=self.attr.__name__)]

        # special case for PyCapsule
        # https://github.com/sizmailov/pybind11-stubgen/issues/86
        if attr_type.__name__ == "PyCapsule" and attr_type.__module__ == "builtins":
            return ["{name}: typing.Any  # PyCapsule()".format(name=self.name)]

        value_lines = repr(self.attr).split("\n")
        typename = self.fully_qualified_name(type(self.attr))

        if len(value_lines) == 1:
            value = value_lines[0]
            # remove random address from <foo.Foo object at 0x1234>
            value = re.sub(r" at 0x[0-9a-fA-F]+>", ">", value)
            if value == "<{typename} object>".format(typename=typename):
                value_comment = ""
            else:
                value_comment = " # value = {value}".format(value=value)
            return [
                "{name}: {typename}{value_comment}".format(
                    name=self.name, typename=typename, value_comment=value_comment
                )
            ]
        else:
            return (
                [
                    "{name}: {typename} # value = ".format(
                        name=self.name, typename=typename
                    )
                ]
                + ['"""']
                + [l.replace('"""', r"\"\"\"") for l in value_lines]
                + ['"""']
            )

    def get_involved_modules_names(self):  # type: () -> Set[str]
        attr_type = type(self.attr)
        if attr_type is type(os):
            return {self.attr.__name__}
        if attr_type.__name__ == "PyCapsule" and attr_type.__module__ == "builtins":
            # PyCapsule rendered as typing.Any
            return {"typing"}
        return {self.attr.__class__.__module__}


class FreeFunctionStubsGenerator(StubsGenerator):
    def __init__(self, name, free_function, module_name):
        self.name = name
        self.member = free_function
        self.module_name = module_name
        self.signatures = []  # type:  List[FunctionSignature]

    def parse(self):
        self.signatures = self.function_signatures_from_docstring(
            self.name, self.member, self.module_name
        )

    def to_lines(self):  # type: () -> List[str]
        result = []
        docstring = self.sanitize_docstring(self.member.__doc__)
        if not docstring and not (
            self.name.startswith("__") and self.name.endswith("__")
        ):
            logger.debug(
                "Docstring is empty for '%s'" % self.fully_qualified_name(self.member)
            )
        for sig in self.signatures:
            if len(self.signatures) > 1:
                result.append("@typing.overload")
            result.append(f"def {sig}")
            if docstring:
                result.append(self.format_docstring(docstring))
                docstring = None  # don't print docstring for other overloads
            else:
                result.append(self.indent("pass"))

        return result

    def get_involved_modules_names(self):  # type: () -> Set[str]
        involved_modules_names = set()
        for s in self.signatures:  # type: FunctionSignature
            for t in s.get_all_involved_types():  # type: str
                module_name = self.module_from_qualname(t)
                if module_name:
                    involved_modules_names.add(module_name)
        return involved_modules_names

DUNDER_METHODS = {
    "__init__",
    "__getitem__",
    "__setitem__",
    "__getstate__",
    "__setstate__",
    "__setattr__",
    "__getattr__",
    "__copy__",
    "__deepcopy__",
    "__iadd__",
    "__idiv__",
    "__imul__",
    "__isub__",
    "__iter__",
}

# always return self
INPLACE_METHODS = {
    "__iadd__",
    "__isub__",
    "__imul__",
    "__idiv__",
}

IGNORE_COMMENTS = {}
# iadd may be inconsistent with add (if add lacks some overrides)
for op in "add", "sub", "mul", "div":
    IGNORE_COMMENTS[f"__i{op}__"] = {"misc"}
# eq/ne may only be implemented for the specific type
for op in "eq", "ne":
    IGNORE_COMMENTS[f"__{op}__"] = {"override"}
# these still return lists, py2 style
for f in "keys", "values", "items":
    IGNORE_COMMENTS[f] = {"override"}
# getitem may be missing SupportsIndex, slice overrides
# iter may return pairs instead of keys for maps
for f in "__getitem__", "__iter__":
    IGNORE_COMMENTS[f] = {"override"}

_container_equivalents: dict[type,type] = {}


class ClassMemberStubsGenerator(FreeFunctionStubsGenerator):
    def __init__(self, name, free_function, class_name, module_name):
        self.class_name = class_name
        super(ClassMemberStubsGenerator, self).__init__(
            name, free_function, module_name
        )

    def to_lines(self):  # type: () -> List[str]
        result = []
        docstring = self.sanitize_docstring(self.member.__doc__)
        if not docstring and not (
            self.name.startswith("__") and self.name.endswith("__")
        ):
            logger.debug(
                "Docstring is empty for '%s'" % self.fully_qualified_name(self.member)
            )
        for sig in self.signatures:
            # detect boost::python self
            if sig._args:
                first = sig._args[0]
                # detect boost::python self
                if first.annotation == self.class_name or sig.name in DUNDER_METHODS:
                    first.name = "self"
                if first.name == "self":
                    first.annotation = None
                else:
                    result.append("@staticmethod")
            # no arg -> staticmethod
            else:
                result.append("@staticmethod")
            
            if sig.name in INPLACE_METHODS:
                sig.rtype = self.class_name

            comment = IGNORE_COMMENTS.get(sig.name, set()).copy()
            if len(self.signatures) > 1:
                result.append("@typing.overload")
                if comment:
                    result[-1] = result[-1] + f" # type: ignore[{','.join(comment)}]"
                    comment = set()
            
            result.append(
                "def {sig}{ellipsis}{comment}".format(
                    sig=sig,
                    ellipsis="" if docstring else " ...",
                    comment=f" # type: ignore[{','.join(comment)}]" if comment else ""
                )
            )
            if docstring:
                result.append(self.format_docstring(docstring))
                docstring = None  # don't print docstring for other overloads
        return result


class PropertyStubsGenerator(StubsGenerator):
    def __init__(self, name, prop, module_name):
        self.name = name
        self.prop = prop
        self.module_name = module_name
        self.signature = None  # type: PropertySignature

    def parse(self):
        self.signature = self.property_signature_from_docstring(
            self.prop, self.module_name
        )

    def to_lines(self):  # type: () -> List[str]

        docstring = self.sanitize_docstring(self.prop.__doc__)
        docstring_prop = "\n\n".join(
            [docstring, ":type: {rtype}".format(rtype=self.signature.rtype)]
        )

        result = [
            "@property",
            "def {field_name}(self) -> {rtype}:".format(
                field_name=self.name, rtype=self.signature.rtype
            ),
            self.format_docstring(docstring_prop),
        ]

        if self.signature.setter_args != "None":
            result.append("@{field_name}.setter".format(field_name=self.name))
            result.append(
                "def {field_name}({args}) -> None:".format(
                    field_name=self.name, args=self.signature.setter_args
                )
            )
            if docstring:
                result.append(self.format_docstring(docstring))
            else:
                result.append(self.indent("pass"))

        return result
    
    def get_involved_modules_names(self):  # type: () -> Set[str]
        involved_modules_names = set()
        for t in self.signature.get_all_involved_types():  # type: str
            module_name = self.module_from_qualname(t)
            if module_name:
                involved_modules_names.add(module_name)
        return involved_modules_names


class ClassStubsGenerator(StubsGenerator):
    ATTRIBUTES_BLACKLIST = (
        "__class__",
        "__module__",
        "__qualname__",
        "__dict__",
        "__weakref__",
        "__annotations__",
        "__instance_size__",
        "__getstate_manages_dict__",
        "__safe_for_unpickling__",
    )
    PYBIND11_ATTRIBUTES_BLACKLIST = ("__entries",)
    METHODS_BLACKLIST = ("__dir__", "__sizeof__", "__setattr__", "__getstate__", "__setstate__", "__copy__", "__deepcopy__", "__str__")
    BASE_CLASS_BLACKLIST = ("pybind11_object", "object")
    CLASS_NAME_BLACKLIST = ("pybind11_type",)

    def __init__(
        self,
        klass,
        attributes_blacklist=ATTRIBUTES_BLACKLIST,
        pybind11_attributes_blacklist=PYBIND11_ATTRIBUTES_BLACKLIST,
        base_class_blacklist=BASE_CLASS_BLACKLIST,
        methods_blacklist=METHODS_BLACKLIST,
        class_name_blacklist=CLASS_NAME_BLACKLIST,
    ):
        self.klass = klass
        assert inspect.isclass(klass)
        assert klass.__name__.isidentifier()

        self.doc_string = None  # type: Optional[str]

        self.classes = []  # type: List[ClassStubsGenerator]
        self.fields = []  # type: List[AttributeStubsGenerator]
        self.properties = []  # type: List[PropertyStubsGenerator]
        self.methods = []  # type: List[ClassMemberStubsGenerator]
        self.alias = []

        self.base_classes = []
        self.involved_modules_names = set()  # Set[str]

        self.attributes_blacklist = attributes_blacklist
        self.pybind11_attributes_blacklist = pybind11_attributes_blacklist
        self.base_class_blacklist = base_class_blacklist
        self.methods_blacklist = methods_blacklist
        self.class_name_blacklist = class_name_blacklist

    def get_involved_modules_names(self):
        return self.involved_modules_names

    @cache
    def get_involved_arg_type_names(self):
        return {t for f in self.methods for sig in f.signatures for t in sig.argtypes.values()}

    def parse(self):
        if self.klass in _visited_objects:
            return
        _visited_objects.append(self.klass)

        bases = inspect.getmro(self.klass)[1:]

        if equivalent := get_container_equivalent(self.klass):
            bases = bases + (equivalent,)

        def is_base_member(name, member):
            for base in bases:
                if hasattr(base, name) and getattr(base, name) is member:
                    return True
            return False

        is_pybind11 = any(base.__name__ == "pybind11_object" for base in bases)

        for name, member in inspect.getmembers(self.klass):
            # check if attribute is in __dict__ (fast path) before slower search in base classes
            if name not in self.klass.__dict__ and is_base_member(name, member):
                continue
            if name.startswith("__pybind11_module"):
                continue
            if (inspect.isroutine(member) or inspect.isclass(member)) and name != member.__name__:
                self.alias.append(AliasStubsGenerator(name, member))
            elif inspect.isroutine(member):
                self.methods.append(
                    ClassMemberStubsGenerator(name, member, self.klass.__qualname__, self.klass.__module__)
                )
            elif name != "__class__" and inspect.isclass(member):
                if (
                    member.__name__ not in self.class_name_blacklist
                    and member.__name__.isidentifier()
                ):
                    self.classes.append(ClassStubsGenerator(member))
            elif isinstance(member, property):
                self.properties.append(
                    PropertyStubsGenerator(name, member, self.klass.__module__)
                )
            elif name == "__doc__":
                self.doc_string = member
            elif not (
                name in self.attributes_blacklist
                or (is_pybind11 and name in self.pybind11_attributes_blacklist)
            ):
                self.fields.append(AttributeStubsGenerator(name, member))
                # logger.warning("Unknown member %s type : `%s` " % (name, str(type(member))))

        # ensure that names/values come after definitions
        if is_boost_python_enum(self.klass):
            ordered = [[], []]
            for p in self.fields:
                ordered[p.name in {"names", "values"}].append(p)
            self.fields = ordered[0] + ordered[1]

        for x in itertools.chain(
            self.classes, self.methods, self.properties, self.fields
        ,
                                 self.alias):
            x.parse()

        for B in bases:
            if (
                B.__name__ != self.klass.__name__
                and B.__name__ not in self.base_class_blacklist
            ):
                self.base_classes.append(B)
                self.involved_modules_names.add(B.__module__)

        for f in self.methods:
            self.involved_modules_names |= f.get_involved_modules_names()
        
        for c in self.classes:
            self.involved_modules_names |= c.get_involved_modules_names()
        
        for prop in self.properties:
            self.involved_modules_names |= prop.get_involved_modules_names()

        for attr in self.fields:
            self.involved_modules_names |= attr.get_involved_modules_names()

        if equivalent:
            class_type_name = self.fully_qualified_name(self.klass)
            if typing.get_origin(equivalent) is Sequence:
                key_type_name, value_type_name = "int", self.fully_qualified_name(typing.get_args(equivalent)[0])
                for f in self.methods:
                    if f.name == "__iter__":
                        f.signatures[0].rtype = f"typing.Iterator[{value_type_name}]"
                    if f.name == "__setitem__":
                        f.signatures[0]._args[1].annotation = "int"
                        f.signatures[0]._args[2].annotation = value_type_name
                    if f.name == "__getitem__":
                        f.signatures[0]._args[1].annotation = "int"
                        f.signatures[0].rtype = value_type_name
                        # add a slice overload, why not
                        if len(f.signatures) == 1:
                            sig = copy.deepcopy(f.signatures[0])
                            sig.rtype = class_type_name
                            sig._args[1].annotation = "slice"
                            f.signatures.append(sig)
                    if f.name == "__delitem__":
                        f.signatures[0]._args[1].annotation = "int"
                    if f.name == "append":
                        f.signatures[0]._args[1].annotation = value_type_name
                    if f.name == "extend":
                        f.signatures[0]._args[1].annotation = f"typing.Iterable[{value_type_name}]"
            if typing.get_origin(equivalent) is Mapping:
                _container_equivalents[self.klass.__item_type__()] = tuple[_type_or_union(self.klass.__key_type__()), _type_or_union(self.klass.__value_type__())]
                key_type_name, value_type_name = (self.fully_qualified_name(n) for n in typing.get_args(equivalent))
                item_type_name = self.fully_qualified_name(self.klass.__item_type__())
                for f in self.methods:
                    if f.name == "__setitem__":
                        f.signatures[0]._args[1].annotation = key_type_name
                        f.signatures[0]._args[2].annotation = value_type_name
                    if f.name == "__getitem__":
                        f.signatures[0]._args[1].annotation = key_type_name
                        f.signatures[0].rtype = value_type_name
                    if f.name == "__delitem__":
                        f.signatures[0]._args[1].annotation = key_type_name
                    if f.name == "__iter__":
                        f.signatures[0].rtype = f"typing.Iterator[{item_type_name}]"
                    if f.name == "itervalues":
                        f.signatures[0]._args[0].name = "self"
                        f.signatures[0].rtype = f"typing.Iterator[{value_type_name}]"
                    if f.name == "iterkeys":
                        f.signatures[0]._args[0].name = "self"
                        f.signatures[0].rtype = f"typing.Iterator[{key_type_name}]"
                    if f.name in ("pop", "get"):
                        f.signatures[0].rtype = value_type_name
                    

    def to_lines(self):  # type: () -> List[str]
        base_classes_list = [
            strip_current_module_name(
                self.fully_qualified_name(b), self.klass.__module__
            )
            for b in self.base_classes
        ]
        result = [
            "class {class_name}({base_classes_list}):{doc_string}".format(
                class_name=self.klass.__name__,
                base_classes_list=", ".join(base_classes_list),
                doc_string="\n" + self.format_docstring(self.doc_string)
                if self.doc_string
                else "",
            ),
        ]
        for cl in self.classes:
            result.extend(map(self.indent, cl.to_lines()))

        for f in self.methods:
            if f.name not in self.methods_blacklist:
                result.extend(map(self.indent, f.to_lines()))

        for p in self.properties:
            result.extend(map(self.indent, p.to_lines()))

        for p in self.fields:
            result.extend(map(self.indent, p.to_lines()))

        result.append(self.indent("pass"))
        return result


class AliasStubsGenerator(StubsGenerator):

    def __init__(self, alias_name, origin):
        self.alias_name = alias_name
        self.origin = origin

    def parse(self):
        pass

    def to_lines(self): # type: () -> List[str]
        return [
            "{alias} = {origin}".format(
                alias=self.alias_name,
                origin=self.fully_qualified_name(self.origin)
            )
        ]

    def get_involved_modules_names(self): # type: () -> Set[str]
        if inspect.ismodule(self.origin):
            return {self.origin.__name__}
        elif inspect.isroutine(self.origin) or inspect.isclass(self.origin):
            return {self.origin.__module__.__name__}
        else:
            return set()


class ModuleStubsGenerator(StubsGenerator):
    CLASS_NAME_BLACKLIST = ClassStubsGenerator.CLASS_NAME_BLACKLIST
    ATTRIBUTES_BLACKLIST = (
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__spec__",
        "__path__",
        "__cached__",
        "__builtins__",
    )

    def __init__(
        self,
        module_or_module_name,
        attributes_blacklist=ATTRIBUTES_BLACKLIST,
        class_name_blacklist=CLASS_NAME_BLACKLIST,
    ):
        if isinstance(module_or_module_name, str):
            self.module = importlib.import_module(module_or_module_name)
        else:
            self.module = module_or_module_name
            assert inspect.ismodule(self.module)

        self.doc_string = None  # type: Optional[str]
        self.classes = []  # type: List[ClassStubsGenerator]
        self.free_functions = []  # type: List[FreeFunctionStubsGenerator]
        self.submodules = []  # type: List[ModuleStubsGenerator]
        self.imported_modules = []  # type: List[str]
        self.imported_classes = {}  # type: Dict[str, type]
        self.attributes = []  # type: List[AttributeStubsGenerator]
        self.alias = []
        self.stub_suffix = ""
        self.write_setup_py = False

        self.attributes_blacklist = attributes_blacklist
        self.class_name_blacklist = class_name_blacklist

    def parse(self):
        if self.module in _visited_objects:
            return
        _visited_objects.append(self.module)
        logger.debug("Parsing '%s' module" % self.module.__name__)
        for name, member in inspect.getmembers(self.module):
            if (inspect.isfunction(member) or inspect.isclass(member)) and name != member.__name__:
                self.alias.append(AliasStubsGenerator(name, member))
            elif inspect.ismodule(member):
                m = ModuleStubsGenerator(member)
                if m.module.__name__.split(".")[:-1] == self.module.__name__.split("."):
                    self.submodules.append(m)
                else:
                    self.imported_modules += [m.module.__name__]
                    logger.debug(
                        "Skip '%s' module while parsing '%s' "
                        % (m.module.__name__, self.module.__name__)
                    )
            elif inspect.isbuiltin(member) or inspect.isfunction(member):
                self.free_functions.append(
                    FreeFunctionStubsGenerator(name, member, self.module.__name__)
                )
            elif inspect.isclass(member):
                if member.__module__ == self.module.__name__:
                    if (
                        member.__name__ not in self.class_name_blacklist
                        and member.__name__.isidentifier()
                    ):
                        self.classes.append(ClassStubsGenerator(member))
                else:
                    self.imported_classes[name] = member
            elif name == "__doc__":
                self.doc_string = member
            elif name not in self.attributes_blacklist:
                self.attributes.append(AttributeStubsGenerator(name, member))

        for x in itertools.chain(
            self.submodules, self.classes, self.free_functions, self.attributes
        ):
            x.parse()

        def class_ordering(
            a, b
        ):  # type: (ClassStubsGenerator, ClassStubsGenerator) -> int
            if a.klass is b.klass:
                return 0
            if issubclass(a.klass, b.klass) or b.klass.__name__ in a.get_involved_arg_type_names():
                return -1
            if issubclass(b.klass, a.klass) or a.klass.__name__ in b.get_involved_arg_type_names():
                return 1
            return 0

        # reorder classes so base classes would be printed before derived
        # and argument types are defined before they are used
        # print([ k.klass.__name__ for k in self.classes ])
        for i in range(len(self.classes)):
            for j in range(i + 1, len(self.classes)):
                if class_ordering(self.classes[i], self.classes[j]) < 0:
                    t = self.classes[i]
                    self.classes[i] = self.classes[j]
                    self.classes[j] = t
        # print( [ k.klass.__name__ for k in self.classes ] )

    def get_involved_modules_names(self):
        result = set(self.imported_modules)

        for attr in self.attributes:
            result |= attr.get_involved_modules_names()

        for C in self.classes:  # type: ClassStubsGenerator
            result |= C.get_involved_modules_names()

        for f in self.free_functions:  # type: FreeFunctionStubsGenerator
            result |= f.get_involved_modules_names()

        return set(result) - {"builtins", "typing", self.module.__name__}

    def to_lines(self):  # type: () -> List[str]

        result = []

        if self.doc_string:
            result += ['"""' + self.doc_string.replace('"""', r"\"\"\"") + '"""']

        if sys.version_info[:2] >= (3, 7):
            result += ["from __future__ import annotations"]

        result += ["import {}".format(self.module.__name__)]

        # import everything from typing
        result += ["import typing"]

        for name, class_ in self.imported_classes.items():
            class_name = getattr(class_, "__qualname__", class_.__name__)
            if name == class_name:
                suffix = ""
            else:
                suffix = " as {}".format(name)
            result += [
                "from {} import {}{}".format(class_.__module__, class_name, suffix)
            ]

        # import used packages
        used_modules = sorted(self.get_involved_modules_names())
        if used_modules:
            # result.append("if TYPE_CHECKING:")
            # result.extend(map(self.indent, map(lambda m: "import {}".format(m), used_modules)))
            result.extend(map(lambda mod: "import {}".format(mod), used_modules))

        if "numpy" in used_modules and not BARE_NUPMY_NDARRAY:
            result += ["_Shape = typing.Tuple[int, ...]"]

        # add space between imports and rest of module
        result += [""]

        globals_ = {}
        exec("from {} import *".format(self.module.__name__), globals_)
        all_ = set(member for member in globals_.keys() if member.isidentifier()) - {
            "__builtins__"
        }
        if all_:
            result.append(
                "__all__ = [\n    "
                + ",\n    ".join(map(lambda s: '"%s"' % s, sorted(all_)))
                + "\n]\n\n"
            )

        for x in itertools.chain(self.classes, self.free_functions, self.attributes,
                                 self.alias):
            result.extend(x.to_lines())
        result.append("")  # Newline at EOF
        return result

    @property
    def short_name(self):
        return self.module.__name__.split(".")[-1]

    def write(self):
        with DirectoryWalkerGuard(self.short_name + self.stub_suffix):
            with open("__init__.pyi", "w", encoding="utf-8") as init_pyi:
                init_pyi.write("\n".join(self.to_lines()))
            for m in self.submodules:
                m.write()

            if self.write_setup_py:
                with open("setup.py", "w", encoding="utf-8") as setuppy:
                    setuppy.write(
                        """from setuptools import setup
import os


def find_stubs(package):
    stubs = []
    for root, dirs, files in os.walk(package):
        for file in files:
            path = os.path.join(root, file).replace(package + os.sep, '', 1)
            stubs.append(path)
    return dict(package=stubs)


setup(
    name='{package_name}-stubs',
    maintainer="{package_name} Developers",
    maintainer_email="example@python.org",
    description="PEP 561 type stubs for {package_name}",
    version='1.0',
    packages=['{package_name}-stubs'],
    # PEP 561 requires these
    install_requires=['{package_name}'],
    package_data=find_stubs('{package_name}-stubs'),
)""".format(
                            package_name=self.short_name
                        )
                    )


def main(args=None):
    parser = ArgumentParser(
        prog="pybind11-stubgen", description="Generates stubs for specified modules"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="the root directory for output stubs",
        default="./stubs",
    )
    parser.add_argument(
        "--root-module-suffix",
        type=str,
        default="-stubs",
        dest="root_module_suffix",
        help="optional suffix to disambiguate from the original package",
    )
    parser.add_argument(
        "--root_module_suffix",
        type=str,
        default=None,
        dest="root_module_suffix_deprecated",
        help="Deprecated.  Use `--root-module-suffix`",
    )
    parser.add_argument("--no-setup-py", action="store_true")
    parser.add_argument(
        "--non-stop", action="store_true", help="Deprecated. Use `--ignore-invalid=all`"
    )
    parser.add_argument(
        "--ignore-invalid",
        nargs="+",
        choices=["signature", "defaultarg", "all"],
        default=[],
        help="Ignore invalid specified python expressions in docstrings",
    )
    parser.add_argument(
        "--skip-signature-downgrade",
        action="store_true",
        help="Do not downgrade invalid function signatures to func(*args, **kwargs)",
    )
    parser.add_argument(
        "--bare-numpy-ndarray",
        action="store_true",
        default=False,
        help="Render `numpy.ndarray` without (non-standardized) bracket-enclosed type and shape info",
    )
    parser.add_argument(
        "-i", "--import-modules", action="append", default=[], metavar="PREIMPORT_MODULES", help="modules to import (but not generate stubs for)"
    )
    parser.add_argument(
        "module_names", nargs="+", metavar="MODULE_NAME", type=str, help="modules names"
    )
    parser.add_argument("--log-level", default="INFO", help="Set output log level")

    sys_args = parser.parse_args(args or sys.argv[1:])

    if sys_args.non_stop:
        sys_args.ignore_invalid = ["all"]
        warnings.warn(
            "`--non-stop` is deprecated in favor of `--ignore-invalid=all`",
            FutureWarning,
        )

    if sys_args.bare_numpy_ndarray:
        global BARE_NUPMY_NDARRAY
        BARE_NUPMY_NDARRAY = True

    if "all" in sys_args.ignore_invalid:
        FunctionSignature.ignore_invalid_signature = True
        FunctionSignature.ignore_invalid_defaultarg = True
    else:
        if "signature" in sys_args.ignore_invalid:
            FunctionSignature.ignore_invalid_signature = True
        if "defaultarg" in sys_args.ignore_invalid:
            FunctionSignature.ignore_invalid_defaultarg = True

    if sys_args.skip_signature_downgrade:
        FunctionSignature.signature_downgrade = False

    if sys_args.root_module_suffix_deprecated is not None:
        sys_args.root_module_suffix = sys_args.root_module_suffix_deprecated
        warnings.warn(
            "`--root_module_suffix` is deprecated in favor of `--root-module-suffix`",
            FutureWarning,
        )

    stderr_handler = logging.StreamHandler(sys.stderr)
    handlers = [stderr_handler]

    logging.basicConfig(
        level=logging.getLevelName(sys_args.log_level),
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )

    for _module_name in sys_args.import_modules:
        importlib.import_module(_module_name)

    with DirectoryWalkerGuard(sys_args.output_dir):
        for _module_name in sys_args.module_names:
            _module = ModuleStubsGenerator(_module_name)
            _module.parse()
            if FunctionSignature.n_fatal_errors() == 0:
                _module.stub_suffix = sys_args.root_module_suffix
                _module.write_setup_py = not sys_args.no_setup_py
                _dir = _module_name.split(".")[:-1] or ["."]
                _dir = os.path.join(*_dir)
                with DirectoryWalkerGuard(_dir):
                    _module.write()

        if FunctionSignature.n_invalid_signatures > 0:
            logger.info("Useful link: Avoiding C++ types in docstrings:")
            logger.info(
                "      https://pybind11.readthedocs.io/en/latest/advanced/misc.html"
                "#avoiding-cpp-types-in-docstrings"
            )

        if FunctionSignature.n_invalid_default_values > 0:
            logger.info("Useful link: Default argument representation:")
            logger.info(
                "      https://pybind11.readthedocs.io/en/latest/advanced/functions.html"
                "#default-arguments-revisited"
            )

        if FunctionSignature.n_fatal_errors() > 0:
            exit(1)


if __name__ == "__main__":
    main()
