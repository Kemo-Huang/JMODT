import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_git_commit_number():
    if not os.path.exists('.git'):
        return '0000000'

    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources],
        # extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
    )
    return cuda_ext


if __name__ == '__main__':
    version = '1.0.0+%s' % get_git_commit_number()

    setup(
        name='jmodt',
        version=version,
        description='JMODT is a codebase for 3D multi-object detection and tracking with camera-LiDAR fusion',
        install_requires=[
            'numpy',
            'torch>=1.5',
            'numba',
            'easydict',
            'pyyaml',
            'scipy',
            'fire',
            'ortools',
            'filterpy',
            'munkres',
            'tqdm',
            'tensorboardX'
        ],
        author='Kemiao Huang',
        author_email='12032943@mail.sustech.edu.cn',
        license='MIT',
        packages=find_packages(exclude=['tools', 'data', 'output']),
        cmdclass={'build_ext': BuildExtension},
        ext_modules=[
            make_cuda_ext(
                name='iou3d_cuda',
                module='jmodt.ops.iou3d',
                sources=[
                    'src/iou3d.cpp',
                    'src/iou3d_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='pointnet2_cuda',
                module='jmodt.ops.pointnet2',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/ball_query_gpu.cu',
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                    'src/interpolate.cpp',
                    'src/interpolate_gpu.cu',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu',
                ]
            ),
            make_cuda_ext(
                name='roipool3d_cuda',
                module='jmodt.ops.roipool3d',
                sources=[
                    'src/roipool3d.cpp',
                    'src/roipool3d_kernel.cu'
                ]
            )
        ],
    )
