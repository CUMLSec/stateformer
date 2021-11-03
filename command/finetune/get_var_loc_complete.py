import json
import sys
import os


def get_data_type_info(f, var, is_arg, count):
    # variable name and type
    varname = var.getName()
    type_object = var.getDataType()
    type_name = type_object.getName()

    # get to what ever the pointer is pointing to
    ptr_bool = False
    for _ in range(type_name.count('*')):
        type_object = type_object.getDataType()
        type_name = type_object.getName()
        ptr_bool = True

    # if a typedef, get the primitive type definition
    try:
        type_object = type_object.getBaseDataType()
        type_name = type_object.getName()
    except:
        pass

    # find if struct, union, enum, or none of the above
    is_struct = False
    is_union = False
    if len(str(type_object).split('\n')) >= 2:
        if 'Struct' in str(type_object).split('\n')[2]:
            is_struct = True
        elif 'Union' in str(type_object).split('\n')[2]:
            is_union = True

    try:
        type_object.getCount()
        is_enum = True
    except:
        is_enum = False

    if ptr_bool:
        type_name += ' *'

    f[varname] = {'type': str(type_name), 'addresses': [],
                  'agg': {'is_enum': is_enum, 'is_struct': is_struct, 'is_union': is_union}}

    locs = ref.getReferencesTo(var)
    for loc in locs:
        f[varname]['addresses'].append(loc.getFromAddress().toString())

    if is_arg:
        # need to store the register the args are saved into.
        f[varname]['register'] = var.getRegister().getName()
        f[varname]['count'] = count

    return f


output_dir = getScriptArgs()[0]

filepath = str(getProgramFile())
filename = filepath.split('/')[-1]

d = {}

getCurrentProgram().setImageBase(toAddr(0), 0)
ref = currentProgram.getReferenceManager()
function = getFirstFunction()

while function is not None:
    funcname = function.name
    d[funcname] = {}
    all_vars = function.getAllVariables()
    all_args = function.getParameters()

    # regular stack vars
    for var in all_vars:
        d[funcname] = get_data_type_info(d[funcname], var, False, -1)

    # function args
    for arg in all_args:
        count = 0
        if arg.getRegister() is not None:
            d[funcname] = get_data_type_info(d[funcname], arg, True, count)
            count += 1

    function = getFunctionAfter(function)

# print(d)
with open(os.path.join(output_dir, filename + '_stacks'), 'w') as f:
    f.write(json.dumps(d))
