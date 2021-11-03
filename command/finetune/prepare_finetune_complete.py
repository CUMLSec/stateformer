import glob
import json
import os
import random
import re
import sys
import argparse

from capstone import *
from elftools.elf.elffile import ELFFile

# from command import params
class params:
    fields = ['static', 'inst_emb', 'inst_pos_emb', 'arch_emb', 'byte1', 'byte2', 'byte3', 'byte4', 'arg_info']

def tokenize(s):
    s = s.replace(',', ' , ')
    s = s.replace('[', ' [ ')
    s = s.replace(']', ' ] ')
    s = s.replace(':', ' : ')
    s = s.replace('*', ' * ')
    s = s.replace('(', ' ( ')
    s = s.replace(')', ' ) ')
    s = s.replace('{', ' { ')
    s = s.replace('}', ' } ')
    s = s.replace('#', '')
    s = s.replace('$', '')
    s = s.replace('!', ' ! ')

    s = re.sub(r'-(0[xX][0-9a-fA-F]+)', r'- \1', s)
    s = re.sub(r'-([0-9a-fA-F]+)', r'- \1', s)

    return s.split()


def get_function_reps(die, mapping):
    functions = []
    for child_die in die.iter_children():

        if child_die.tag.split('_')[-1] == 'subprogram':
            function = {}
            try:
                function['start_addr'] = child_die.attributes['DW_AT_low_pc'][2]
                function['end_addr'] = function['start_addr'] + child_die.attributes['DW_AT_high_pc'][2]
                function['name'] = child_die.attributes['DW_AT_name'][2].decode('utf-8')
                functions.append(function)
            except KeyError:
                continue

    return functions


def get_type(type_str, agg):

    if '*' in type_str:
        return get_type(type_str.replace('*', ''), agg)+'*'
    elif '[' in type_str and ']' in type_str:
        return 'array'
    elif agg['is_enum']:
        return 'enum'
    elif agg['is_struct']:
        return 'struct'
    elif agg['is_union']:
        return 'union'
    elif 'void' in type_str:
        return 'void'

    elif 'float' in type_str:
        return 'float'
    elif 'long' in type_str and 'double' in type_str:
        return 'long double'
    elif 'double' in type_str:
        return 'double'

    elif 'char' in type_str:
        if 'u' in type_str:
            return 'unsigned char'
        return 'signed char'
    elif 'short' in type_str:
        if 'u' in type_str:
            return 'unsigned short'
        return 'signed short'
    elif 'int' in type_str:
        if 'u' in type_str:
            return 'unsigned int'
        return 'signed int'
    elif 'longlong' in type_str:
        if 'u' in type_str:
            return 'unsigned long long'
        return 'signed long long'
    elif 'long' in type_str:
        if 'u' in type_str:
            return 'unsigned long'
        return 'signed long'

    elif 'undefined' in type_str:
        return 'undefined'

    print(type_str)
    return '?you shouldnt be seeing this?'


def test_hex(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False


def get_reg(tokens):
    if tokens[-1] == ']' or test_hex(tokens[-1]):
        register = tokens[1].upper()
    else:
        register = tokens[-1].upper()
    return register


# gets the type of an instruction that has a stack xref
def get_ds_loc(loc_dict, address, funcname):
    for var in loc_dict[funcname]:
        if address in [int(i, 16) for i in loc_dict[funcname][var]['addresses']]:
            return get_type(loc_dict[funcname][var]['type'], loc_dict[funcname][var]['agg'])
    return 'no-access'


# gets the type of an argument using the register name where it's stored
def get_arg_stack_loc(loc_dict, register, funcname):
    for var in loc_dict[funcname]:
        if ('register' in loc_dict[funcname][var] 
            and register == loc_dict[funcname][var]['register']):
            return get_type(loc_dict[funcname][var]['type'], loc_dict[funcname][var]['agg'])
    return 'undefined'


# gets overall argument info for each function
def get_arg_info(loc_dict, funcname):
    arg_list = []
    for var in loc_dict[funcname]:
        if 'register' in loc_dict[funcname][var].keys():
            arg_list.append((loc_dict[funcname][var]['count'], get_type(loc_dict[funcname][var]['type'], loc_dict[funcname][var]['agg'])))
    arg_list.sort()
    leng = str(len(arg_list))

    while len(arg_list) < 3:
        arg_list.append('##')
    arg_list = [arg_type for (order, arg_type) in arg_list]

    return [leng] + arg_list[:3]


def hex2str(s, b_len=8):
    num = s.replace('0x', '')

    # handle 64-bit cases, we choose the lower 4 bytes, thus 8 numbers
    if len(num) > b_len:
        num = num[-b_len:]

    num = '0' * (b_len - len(num)) + num
    return num


def byte2seq(value_list):
    return [value_list[i:i + 2] for i in range(len(value_list) - 2)]


parser = argparse.ArgumentParser(description='Output ground truth')
parser.add_argument('--output_dir', type=str, nargs=1,
                    help='where ground truth is output')
parser.add_argument('--input_dir', type=str, nargs=1,
                    help='directory where input binaries are')
parser.add_argument('--stack_dir', type=str, nargs=1,
                    help='where the stack files are (same as argument for get_var_loc.py')
parser.add_argument('--arch', type=str, nargs=1,
                    help='architecture of binary')

args = parser.parse_args()

output_dir = args.output_dir[0]
input_dir = args.input_dir[0]
stack_dir = args.stack_dir[0]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

train_file = {field: open(os.path.join(output_dir, f'train.{field}'), 'w') for field in params.fields}
valid_file = {field: open(os.path.join(output_dir, f'valid.{field}'), 'w') for field in params.fields}

train_label = open(os.path.join(output_dir, 'train.label'), 'w')
valid_label = open(os.path.join(output_dir, 'valid.label'), 'w')

file_list = glob.glob(os.path.join(input_dir, '*'), recursive=True)
# filename = 'command/ghidra/ds_test_dwarf'

for filename in file_list:
    # load data structure information from ghidra
    with open(os.path.join(stack_dir, f'{os.path.basename(filename)}_stacks'), 'r') as f:
        loc_dict = json.loads(f.read())

    with open(filename, 'rb') as f:
        elffile = ELFFile(f)
        dwarf = elffile.get_dwarf_info()

        # disassemble the byte code with capstone
        code = elffile.get_section_by_name('.text')
        opcodes = code.data()
        addr = code['sh_addr']
        md = Cs(CS_ARCH_X86, CS_MODE_64)

        for CU in dwarf.iter_CUs():
            function_reps = get_function_reps(CU.get_top_DIE(), None)

            for func in function_reps:
                start_addr = func['start_addr']
                end_addr = func['end_addr']

                func_args = {}
                used_regs = set()

                # input
                static = []
                inst_pos = []
                op_pos = []
                arch = []
                byte1 = []
                byte2 = []
                byte3 = []
                byte4 = []

                # output
                labels = []

                inst_pos_counter = 0

                try:
                    for address, size, op_code, op_str in md.disasm_lite(opcodes, addr):

                        if start_addr <= address < end_addr:
                            tokens = tokenize(f'{op_code} {op_str}')
                            label = get_ds_loc(loc_dict, address, func['name'])

                            # get the register and stack location for likely arg vars from the 
                            # op_str and label the instruction by using the register->param type
                            # mapping from Ghidra. A mapping of stack location -> type is stored
                            # for whenever else the location is seen.
                            if label == 'undefined' and '[' in tokens and op_code == 'mov':
                                reg = get_reg(tokens)

                                loc = op_str[op_str.find("[")+1:op_str.find("]")]
                                if loc in func_args:
                                    label = func_args[loc]

                                else:
                                    label = get_arg_stack_loc(loc_dict, reg, func['name'])
                                    func_args[loc] = label

                            for i, token in enumerate(tokens):
                                if '0x' in token.lower():
                                    static.append('hexvar')
                                    bytes = byte2seq(hex2str(token.lower()))
                                    byte1.append(bytes[0])
                                    byte2.append(bytes[1])
                                    byte3.append(bytes[2])
                                    byte4.append(bytes[3])

                                elif token.lower().isdigit():
                                    static.append('num')
                                    bytes = byte2seq(hex2str(hex(int(token.lower()))))
                                    byte1.append(bytes[0])
                                    byte2.append(bytes[1])
                                    byte3.append(bytes[2])
                                    byte4.append(bytes[3])
                                    
                                else:
                                    static.append(token)
                                    byte1.append('##')
                                    byte2.append('##')
                                    byte3.append('##')
                                    byte4.append('##')

                                inst_pos.append(str(inst_pos_counter))
                                op_pos.append(str(i))
                                arch.append(args.arch[0])

                                labels.append(label)

                            inst_pos_counter += 1

                            # print(str(address) + "\t"+ label+ "\t"+ op_code + "\t"+ op_str )

                except CsError as e:
                    print("ERROR: %s" % e)


                arg_info = get_arg_info(loc_dict, func['name'])

                # skip invalid functions
                if len(labels) < 30 or len(labels) > 510 or len(set(labels)) == 1:
                    continue

                if not random.random() < 0.1:
                    train_file[params.fields[0]].write(' '.join(static) + '\n')
                    train_file[params.fields[1]].write(' '.join(inst_pos) + '\n')
                    train_file[params.fields[2]].write(' '.join(op_pos) + '\n')
                    train_file[params.fields[3]].write(' '.join(arch) + '\n')
                    train_file[params.fields[4]].write(' '.join(byte1) + '\n')
                    train_file[params.fields[5]].write(' '.join(byte2) + '\n')
                    train_file[params.fields[6]].write(' '.join(byte3) + '\n')
                    train_file[params.fields[7]].write(' '.join(byte4) + '\n')
                    train_file[params.fields[8]].write(' '.join(arg_info) + '\n')

                    train_label.write(' '.join(labels) + '\n')


                else:
                    valid_file[params.fields[0]].write(' '.join(static) + '\n')
                    valid_file[params.fields[1]].write(' '.join(inst_pos) + '\n')
                    valid_file[params.fields[2]].write(' '.join(op_pos) + '\n')
                    valid_file[params.fields[3]].write(' '.join(arch) + '\n')
                    valid_file[params.fields[4]].write(' '.join(byte1) + '\n')
                    valid_file[params.fields[5]].write(' '.join(byte2) + '\n')
                    valid_file[params.fields[6]].write(' '.join(byte3) + '\n')
                    valid_file[params.fields[7]].write(' '.join(byte4) + '\n')
                    valid_file[params.fields[8]].write(' '.join(arg_info) + '\n')

                    valid_label.write(' '.join(labels) + '\n')

for k in train_file:
    train_file[k].close()
for k in valid_file:
    valid_file[k].close()
train_label.close()
valid_label.close()
