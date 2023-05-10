"""
>>> sig = "__init__( (object)arg1 [, (int)photonsPerStep=200 [, (int)highPhotonsPerStep=0 [, (float)useHighPhotonsPerStepStartingFromNumPhotons=1000000000.0]]]) -> None :"
>>> transform_signatures(sig)
'1. __init__(arg1: object, photonsPerStep: int=200, highPhotonsPerStep: int=0, useHighPhotonsPerStepStartingFromNumPhotons: float=1000000000.0) -> None:'

>>> sig = "__init__( (object)arg1, (I3CLSimOpenCLDevice)device, (int)workgroupSize, (int)workItemsPerIteration, (int)iterations, (icecube._phys_services.I3RandomService)randomService [, (str)rngType='']) -> None :"
>>> transform_signatures(sig)
"1. __init__(arg1: object, device: I3CLSimOpenCLDevice, workgroupSize: int, workItemsPerIteration: int, iterations: int, randomService: icecube._phys_services.I3RandomService, rngType: str='') -> None:"

>>> sig = "__init__( (object)arg1, (I3CLSimOpenCLDevice)device, (int)workgroupSize, (int)workItemsPerIteration, (icecube._phys_services.I3RandomService)randomService, (object)randomDistribution [, (icecube._dataclasses.ListDouble)runtimeParameters=[]]) -> None :"
>>> transform_signatures(sig)
'1. __init__(arg1: object, device: I3CLSimOpenCLDevice, workgroupSize: int, workItemsPerIteration: int, randomService: icecube._phys_services.I3RandomService, randomDistribution: object, runtimeParameters: icecube._dataclasses.ListDouble=[]) -> None:'

>>> sig = "__init__( (object)arg1, (icecube._simclasses.I3CLSimFunction)flasherSpectrumNoBias, (I3CLSimSpectrumTable)spectrumTable, (object)angularProfileDistributionPolar, (object)angularProfileDistributionAzimuthal, (object)timeDelayDistribution [, (bool)interpretAngularDistributionsInPolarCoordinates=False [, (int)photonsPerStep=400]]) -> None:"
>>> transform_signatures(sig)
'1. __init__(arg1: object, flasherSpectrumNoBias: icecube._simclasses.I3CLSimFunction, spectrumTable: I3CLSimSpectrumTable, angularProfileDistributionPolar: object, angularProfileDistributionAzimuthal: object, timeDelayDistribution: object, interpretAngularDistributionsInPolarCoordinates: bool=False, photonsPerStep: int=400) -> None:'

>>> sig = "initializeOpenCL( (I3CLSimOpenCLDevice)openCLDevice, (icecube._phys_services.I3RandomService)randomService, (icecube._simclasses.I3SimpleGeometry)geometry, (icecube._simclasses.I3CLSimMediumProperties)mediumProperties, (icecube._simclasses.I3CLSimFunction)wavelengthGenerationBias, (icecube._simclasses.I3CLSimRandomValuePtrSeries)wavelengthGenerators [, (bool)enableDoubleBuffering=False [, (bool)doublePrecision=False [, (bool)stopDetectedPhotons=True [, (bool)saveAllPhotons=False [, (float)saveAllPhotonsPrescale=0.01 [, (float)fixedNumberOfAbsorptionLengths=nan [, (float)pancakeFactor=1.0 [, (int)photonHistoryEntries=0 [, (int)limitWorkgroupSize=0]]]]]]]]]) -> I3CLSimStepToPhotonConverterOpenCL :"
>>> transform_signatures(sig)
"1. initializeOpenCL(openCLDevice: I3CLSimOpenCLDevice, randomService: icecube._phys_services.I3RandomService, geometry: icecube._simclasses.I3SimpleGeometry, mediumProperties: icecube._simclasses.I3CLSimMediumProperties, wavelengthGenerationBias: icecube._simclasses.I3CLSimFunction, wavelengthGenerators: icecube._simclasses.I3CLSimRandomValuePtrSeries, enableDoubleBuffering: bool=False, doublePrecision: bool=False, stopDetectedPhotons: bool=True, saveAllPhotons: bool=False, saveAllPhotonsPrescale: float=0.01, fixedNumberOfAbsorptionLengths: float=float('nan'), pancakeFactor: float=1.0, photonHistoryEntries: int=0, limitWorkgroupSize: int=0) -> I3CLSimStepToPhotonConverterOpenCL:"

>>> sig = 'get( (I3CLSimFunctionMap)arg1, (icecube._icetray.OMKey)arg2 [, (object)default_val]) -> object :'
>>> transform_signatures(sig)
'1. get(arg1: I3CLSimFunctionMap, arg2: icecube._icetray.OMKey) -> object:\\n2. get(arg1: I3CLSimFunctionMap, arg2: icecube._icetray.OMKey, default_val: object) -> object:'

>>> transform_signatures("__init__( (object)arg1 [, (float)salinity=0.03844 [, (float)temperature=13.1 [, (float)pressure=1.517810339670313e+17 [, (float)n0=1.31405 [, (float)n1=1.45e-05 [, (float)n2=0.0001779 [, (float)n3=1.05e-06 [, (float)n4=1.6e-08 [, (float)n5=2.02e-06 [, (float)n6=15.868 [, (float)n7=0.01155 [, (float)n8=0.00423 [, (float)n9=4382.0 [, (float)n10=1145500.0]]]]]]]]]]]]]]) -> None :")
'1. __init__(arg1: object, salinity: float=0.03844, temperature: float=13.1, pressure: float=1.517810339670313e+17, n0: float=1.31405, n1: float=1.45e-05, n2: float=0.0001779, n3: float=1.05e-06, n4: float=1.6e-08, n5: float=2.02e-06, n6: float=15.868, n7: float=0.01155, n8: float=0.00423, n9: float=4382.0, n10: float=1145500.0) -> None:'

>>> transform_signatures("__init__( (object)arg1, (icecube._dataclasses.I3Geometry)arg2 [, (I3ScaleCalculator.IceCubeConfig)arg3 [, (I3ScaleCalculator.IceTopConfig)arg4 [, (icecube._dataclasses.ListInt)arg5 [, (icecube._dataclasses.ListInt)arg6 [, (int)arg7 [, (int)arg8]]]]]]) -> None:")
'1. __init__(arg1: object, arg2: icecube._dataclasses.I3Geometry) -> None:\\n2. __init__(arg1: object, arg2: icecube._dataclasses.I3Geometry, arg3: I3ScaleCalculator.IceCubeConfig) -> None:\\n3. __init__(arg1: object, arg2: icecube._dataclasses.I3Geometry, arg3: I3ScaleCalculator.IceCubeConfig, arg4: I3ScaleCalculator.IceTopConfig) -> None:\\n4. __init__(arg1: object, arg2: icecube._dataclasses.I3Geometry, arg3: I3ScaleCalculator.IceCubeConfig, arg4: I3ScaleCalculator.IceTopConfig, arg5: icecube._dataclasses.ListInt) -> None:\\n5. __init__(arg1: object, arg2: icecube._dataclasses.I3Geometry, arg3: I3ScaleCalculator.IceCubeConfig, arg4: I3ScaleCalculator.IceTopConfig, arg5: icecube._dataclasses.ListInt, arg6: icecube._dataclasses.ListInt) -> None:\\n6. __init__(arg1: object, arg2: icecube._dataclasses.I3Geometry, arg3: I3ScaleCalculator.IceCubeConfig, arg4: I3ScaleCalculator.IceTopConfig, arg5: icecube._dataclasses.ListInt, arg6: icecube._dataclasses.ListInt, arg7: int) -> None:\\n7. __init__(arg1: object, arg2: icecube._dataclasses.I3Geometry, arg3: I3ScaleCalculator.IceCubeConfig, arg4: I3ScaleCalculator.IceTopConfig, arg5: icecube._dataclasses.ListInt, arg6: icecube._dataclasses.ListInt, arg7: int, arg8: int) -> None:'

>>> transform_signatures("get( (AntennaSpectrumMap)arg1, (AntennaKey)arg2 [, (object)default_val]) -> object :")
'1. get(arg1: AntennaSpectrumMap, arg2: AntennaKey) -> object:\\n2. get(arg1: AntennaSpectrumMap, arg2: AntennaKey, default_val: object) -> object:'
"""

import pyparsing as pp


def make_signature() -> pp.ParserElement:
    """
    A grammar for parsing signatures generated by boost-python
    """
    name = pp.Word(pp.alphas + "_", pp.alphanums + "_")
    qualname = pp.Combine(pp.delimited_list(name, "."), join_string=".")

    expr = pp.Forward()

    singletons = pp.Literal("True") | pp.Literal("False") | pp.Literal("None")

    plusminus = pp.Optional(pp.Or(list("+-")))
    # a usuable repr for special float values
    denormals = pp.Literal("nan") | pp.Combine(plusminus + pp.Literal("inf"))
    denormals.set_parse_action(lambda s, loc, tok: f"float('{tok[0]}')")

    integer = pp.Word(pp.nums)
    number = pp.Combine(
        plusminus
        + integer
        + pp.Opt(pp.Literal(".") + integer)
        + pp.Opt(pp.Or(list("eE")) + plusminus + integer)
    )

    literal = pp.quoted_string | number | singletons | denormals
    arg_list = pp.delimited_list(
        pp.Group(expr), pp.Literal(",") + pp.ZeroOrMore(pp.White()), combine=True
    )
    function_call = qualname + pp.Literal("(") + pp.Opt(arg_list) + pp.Literal(")")
    literal_list = pp.Literal("[") + pp.Opt(arg_list) + pp.Literal("]")

    expr <<= pp.Combine(literal | literal_list | function_call | qualname)

    annotation = (
        pp.Suppress("(") + qualname.set_results_name("annotation") + pp.Suppress(")")
    )

    parameter = (
        annotation
        + name.set_results_name("arg")
        + pp.Opt(pp.Suppress("=") + expr, default=None).set_results_name("default")
    )

    parameter_run = pp.delimited_list(pp.Group(parameter), ", ")

    optional_parameters = pp.nested_expr("[, ", "]", content=parameter_run)

    parameters = pp.Opt(parameter_run, default=[]).set_results_name(
        "required_args"
    ) + pp.Opt(optional_parameters, default=[]).set_results_name("optional_args")

    rettype = pp.Opt(pp.Suppress("->") + qualname, default="Any")

    function_def = (
        name.set_results_name("name")
        + pp.Suppress("(")
        + parameters
        + pp.Suppress(")")
        + rettype.set_results_name("rettype")
        + pp.Opt(pp.Suppress(":"))
    )

    return function_def


FunctionDef = make_signature()

parse = make_signature().parse_string


def format_arg(arg: pp.ParseResults):
    default = arg.default.as_list()[0] if arg.default else None
    return f"{arg.arg}: {arg.annotation}" + ("" if default is None else f"={default}")


def format_args(tokens: pp.ParseResults):
    return ", ".join(format_arg(arg) for arg in tokens)


def expand_optional_args(tokens: pp.ParseResults):
    args = []
    while True:
        arg = tokens[0]
        default = arg.default.as_list()[0] if arg.default else None
        if default is None:
            yield args
        args.append(arg)            
        if len(tokens) > 1:
            tokens = tokens[1]
        else:
            yield args
            break

def format_signature(tokens: pp.ParseResults):
    if tokens.optional_args:
        for num, args in enumerate(expand_optional_args(tokens.optional_args[0]), 1):
            yield f"{num}. {tokens.name}({format_args(tokens.required_args)}{', ' if args else ''}{format_args(args)}) -> {tokens.rettype[0]}:"
    else:
        yield f"1. {tokens.name}({format_args(tokens.required_args)}) -> {tokens.rettype[0]}:"


def transform_signatures(doc: str):
    """
    Transform signatures in a boost-python-generated docstring to PEP 484
    """
    pos = 0
    out = ""
    for tokens, start, end in FunctionDef.scan_string(doc):
        out += doc[pos:start]
        out += "\n".join(format_signature(tokens))
        out += "\n"
        pos = end
    out += doc[pos:]
    return out.rstrip()
