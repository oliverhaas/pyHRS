from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

include_dirs = [numpy.get_include()]
library_dirs = []
include_dirs.append("/usr/local/include/")
library_dirs.append("/usr/local/lib/")


setup(
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension("transferMap", ["transferMap.pyx"],
  		                       include_dirs=include_dirs),
                     Extension("dedx", ["dedx.pyx"],
                               include_dirs=include_dirs),
                     Extension("constants", ["constants.pyx"],
				               include_dirs=include_dirs,
				               library_dirs=library_dirs),
                     Extension("specfun", ["specfun.pyx"],
				               include_dirs=include_dirs,
				               library_dirs=library_dirs,
				               extra_link_args=['-lgmp','-lmpfr'],
				               extra_compile_args=['-lgmp','-lmpfr']),  
                     Extension("randomGen", ["randomGen.pyx"],
                      		   include_dirs=include_dirs,
                      		   library_dirs=library_dirs,
                               extra_link_args=['-lgsl', '-lgslcblas'],
                               extra_compile_args=['-lgsl', '-lgslcblas']),
                     Extension("spline", ["spline.pyx"],
                               include_dirs=include_dirs,
                               library_dirs=library_dirs),
                     Extension("util", ["util.pyx"],
                               include_dirs=include_dirs,
                               library_dirs=library_dirs),
                     Extension("fieldMap", ["fieldMap.pyx"],
                               include_dirs=include_dirs,
                               library_dirs=library_dirs)
                           
					]
)

