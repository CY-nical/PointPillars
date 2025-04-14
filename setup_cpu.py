from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='pointpillars',
    version='0.1',
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name='pointpillars.ops.voxel_op',
            sources=[
                'pointpillars/ops/voxelization/voxelization.cpp',
                'pointpillars/ops/voxelization/voxelization_cpu.cpp',
            ]
        ),
        # CppExtension(
        #     name='pointpillars.ops.iou3d_op',
        #     sources=[
        #         'pointpillars/ops/iou3d/iou3d.cpp',
        #     ]
        # )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False
)