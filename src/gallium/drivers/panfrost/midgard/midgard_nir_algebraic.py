#
# Copyright (C) 2016 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

import argparse
import sys
import math

a = 'a'
b = 'b'

algebraic = [
    (('f2b', a), ('fne', a, 0.0)),
    (('isign', a), ('imin', ('imax', a, -1), 1)),
    (('fge', a, b), ('flt', b, a)),

    # XXX: We have hw ops for this, just unknown atm..
    #(('fsign@32', a), ('i2f32@32', ('isign', ('f2i32@32', ('fmul', a, 0x43800000)))))
    #(('fsign', a), ('fcsel', ('fge', a, 0), 1.0, ('fcsel', ('flt', a, 0.0), -1.0, 0.0)))
    (('fsign', a), ('bcsel', ('fge', a, 0), 1.0, -1.0)),

    (('b2i@32', a), ('iand', a, 1))
]

# Midgard scales fsin/fcos arguments by pi.
# Pass must be run only once, after the main loop

scale_trig = [
        (('fsin', a), ('fsin', ('fdiv', a, math.pi))),
        (('fcos', a), ('fcos', ('fdiv', a, math.pi))),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--import-path', required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.import_path)
    run()


def run():
    import nir_algebraic  # pylint: disable=import-error

    print '#include "midgard_nir.h"'
    print nir_algebraic.AlgebraicPass("midgard_nir_lower_algebraic",
                                      algebraic).render()

    print nir_algebraic.AlgebraicPass("midgard_nir_scale_trig",
                                      scale_trig).render()


if __name__ == '__main__':
    main()
