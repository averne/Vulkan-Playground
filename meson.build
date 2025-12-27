project('Vulkan-testbed', 'cpp',
    default_options: ['cpp_std=c++23']
)

if get_option('buildtype') == 'debug'
    add_global_arguments('-DDEBUG', language: 'cpp')
endif

glsllang = find_program('glslangValidator')
shader = custom_target('shader',
    command : [glsllang, '-V', '@INPUT@', '--target-env', 'vulkan1.2', '-o',  '@OUTPUT@'],
    input : 'src/prefix_scan.comp',
    output : '@BASENAME@.spv',
)

executable('vk-test', 'src/main.cpp',
    dependencies: [dependency('vulkan', required: true), declare_dependency(sources: shader)],
)
