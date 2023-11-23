import os
import re
import sys
import sysconfig
import platform
import subprocess
import multiprocessing

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

from datetime import datetime
from sysconfig import get_config_var


if platform.architecture()[0] == "32bit":
    raise SystemExit("Requires 64 bit Python")

cfg = 'Debug' if "--debug" in sys.argv else 'Release'
ode_dir = os.path.abspath("MoCCASimuBackend")
system_type = platform.system()

class CythonCMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake is required.")

        self.cmake_build(ode_dir)
        super(CythonCMakeBuild, self).run()
        self.copy_extensions_to_source()

    def cmake_build(self, cmake_path: str):
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + ode_dir]
        build_args = ['--config', cfg]

        if system_type == "Windows":
            cmake_args += ['-DCMAKE_INSTALL_PREFIX={}'.format(ode_dir)]
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())

        subprocess.check_call(['cmake', cmake_path] + cmake_args, cwd=ode_dir, env=env)
        num_cores = multiprocessing.cpu_count()
        subprocess.check_call(['cmake', '--build', ode_dir, '-j', str(num_cores)] + build_args, cwd=ode_dir)
        print()


def _prepare_ModifiedODE_ext():
    import numpy as np
    from Cython.Build import cythonize
    
    proj_dir = os.path.abspath(os.path.join(ode_dir, ".."))
    
    ModifiedODE_cflags = ['-DdDOUBLE']
    if system_type == "Windows":
        ModifiedODE_link_args = ['/LIBPATH:' + os.path.join(ode_dir, cfg),
                                'ModifiedODE.lib', 'MotionUtils.lib', 'DrawStuff.lib', 'user32.lib', 'gdi32.lib']
        ModifiedODE_link_args.append('GlU32.Lib')
        ModifiedODE_link_args.append('OpenGL32.Lib')
        ModifiedODE_link_args.append(os.path.join(proj_dir, 'ThirdParty', 'glew32', 'glew32.lib'))
        
    elif system_type == 'Linux':  # WSL 1 of Ubuntu 20.04 also returns 'Linux'
        ModifiedODE_link_args = [
            # '-Wl,--no-undefined', 
            'libModifiedODE.a', 'libMotionUtils.a','libDrawStuff.a',
            '-L%s'%get_config_var('LIBPL'),
            '-lpython%s'%get_config_var('LDVERSION'),
            '-L%s'%os.path.join(ode_dir)
        ]

    elif system_type == 'Darwin': # mac
        ModifiedODE_link_args = [
            # '-Wl,--no-undefined', 
            os.path.join(ode_dir,'libModifiedODE.a'), 
            os.path.join(ode_dir,'libMotionUtils.a'),
            os.path.join(ode_dir,'libDrawStuff.a'),
            '-L%s'%get_config_var('LIBPL'),
            '-lpython%s'%get_config_var('LDVERSION'),
            '-L%s'%os.path.join(ode_dir)
        ]

    else:
        raise NotImplementedError(system_type)


    ModifiedODE_ext = Extension("*", [os.path.join(ode_dir, "*.pyx")], 
                            include_dirs=[np.get_include(), 
                                            os.path.join(ode_dir, "include"),
                                            os.path.join(ode_dir, "src"),
                                            os.path.join(os.path.join(ode_dir, "src"), "joints"),
                                            os.path.join(ode_dir, "Utils"),
                                            os.path.join(os.path.join(proj_dir, "ThirdParty"), "eigen-3.3.8"),
                                            os.path.join(proj_dir, "ThirdParty")],
                            extra_compile_args=ModifiedODE_cflags,
                            extra_link_args=ModifiedODE_link_args,
                            language="c++",
                            )
    
    return cythonize(ModifiedODE_ext)

if __name__ == "__main__":
    now = datetime.now()
    setup(
        name="MoCCASimuBackend",
        version="0.12" + now.strftime(".%Y.%m.%d.%H.%M.%S"),
        author="PKU-MoCCA",
        # author_email="",
        # maintainer="",
        # maintainer_email="",
        # url="",
        description="A simple simulation framework based on a modified version of ODE",
        # long_description="""""",
        classifiers=[
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Physics'            
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Cython',
        ],        
        platforms=["Windows", "Linux", "Mac OS-X", "Unix"],        
        license="BSD License, GNU Lesser General Public License (LGPL)",        
        
        install_requires=[ 'numpy', 'scipy' ],
        setup_requires=['cython', 'numpy'],
        
        cmdclass={"build_ext": CythonCMakeBuild},
        
        ext_modules=_prepare_ModifiedODE_ext()
        )
