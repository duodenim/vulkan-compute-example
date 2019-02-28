#[macro_use]
extern crate ash;
use ash::Entry;
use ash::vk;
use ash::version::EntryV1_0;
use ash::version::InstanceV1_0;
use ash::version::DeviceV1_0;
use ash::extensions::ext::DebugReport;
use ash::extensions::khr::Swapchain;
use ash::extensions::khr::Surface;
use ash::vk::Handle;
use std::ffi::CString;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::os::raw::c_void;

extern crate byteorder;
use byteorder::{NativeEndian, ByteOrder};

extern crate sdl2;
use sdl2::event::Event;

extern crate vk_mem;

//Number of particles to run, must match with NUM_PARTICLES in vertex and compute shaders
const NUM_PARTICLES: u32  = 400;

//Reserve space for particles plus an extra vec4 for passing global params (time)
const BUFFER_SIZE: u64 = ((NUM_PARTICLES + 1) * 4 * std::mem::size_of::<f32>() as u32) as u64;

const WINDOW_SIZE_X: u32 = 640;
const WINDOW_SIZE_Y: u32 = 480;

unsafe extern "system" fn vulkan_debug_callback(
    _: vk::DebugReportFlagsEXT,
    _: vk::DebugReportObjectTypeEXT,
    _: u64,
    _: usize,
    _: i32,
    _: *const c_char,
    p_message: *const c_char,
    _: *mut c_void
) -> u32 {
    println!("{:?}", CStr::from_ptr(p_message));
    vk::FALSE
}

fn main() {

    //Setup SDL window
    let sdl_context = sdl2::init().unwrap();
    let video_context = sdl_context.video().unwrap();
    let window = video_context.window("demo", WINDOW_SIZE_X, WINDOW_SIZE_Y).vulkan().build().unwrap();

    //Get required Vulkan instance extensions for a Vulkan surface
    let sdl_vk_exts = window.vulkan_instance_extensions().unwrap();

    //Get vulkan function entrypoints
    let entry = Entry::new().unwrap();

    //Create Vulkan instance
    let instance  = {
        let app_name = CString::new("AshTest").unwrap();
        let layer_names = [CString::new("VK_LAYER_LUNARG_standard_validation").unwrap()];
        let layer_names_raw: Vec<*const i8> = layer_names.iter().map(|name| name.as_ptr()).collect();
        let mut extension_names = vec![DebugReport::name().as_ptr()];

        for ext in sdl_vk_exts.iter() {
            extension_names.push(ext.as_ptr() as *const i8);
        }
        let appinfo = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(0)
            .engine_name(&app_name)
            .engine_version(0)
            .api_version(vk_make_version!(1, 0, 0));
        let inst_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&appinfo)
            .enabled_layer_names(&layer_names_raw)
            .enabled_extension_names(&extension_names);

        unsafe { entry.create_instance(&inst_create_info, None).unwrap() }
    };
    
    //Create a debugging callback function for error handling
    let debug_info = vk::DebugReportCallbackCreateInfoEXT::builder()
        .flags(vk::DebugReportFlagsEXT::ERROR | vk::DebugReportFlagsEXT::WARNING | vk::DebugReportFlagsEXT::PERFORMANCE_WARNING)
        .pfn_callback(Some(vulkan_debug_callback));

    let debug_report_loader = DebugReport::new(&entry, &instance);
    let debug_call_back = unsafe { debug_report_loader.create_debug_report_callback(&debug_info, None).unwrap() };

    //Print out information about available Vulkan devices
    let pdevices = unsafe { instance.enumerate_physical_devices().unwrap() };
    println!("Available devices:");
    for pdev in pdevices.iter() {
        let properties = unsafe { instance.get_physical_device_properties(*pdev) };
        let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) };
        println!("{:?}", name);
        println!("{:?}", properties.limits.point_size_range);
    }

    //Select the first available physical device
    let physical_device = pdevices[0];

    //Create surface (parameters are passed around as raw handles as the surface creation is handled in SDL instead of ash)
    let inst_handle = instance.handle().as_raw() as usize;
    let surface_ext = Surface::new(&entry, &instance);
    let surface: vk::SurfaceKHR = vk::Handle::from_raw(window.vulkan_create_surface(inst_handle).unwrap());
    let _surface_caps = unsafe { surface_ext.get_physical_device_surface_capabilities(physical_device, surface).unwrap() };
    let surface_formats = unsafe { surface_ext.get_physical_device_surface_formats(physical_device, surface).unwrap() };
    let _surface_present_modes = unsafe { surface_ext.get_physical_device_surface_present_modes(physical_device, surface).unwrap() };

    //Device queues are grouped into families with support for different operations
    //This app needs a queue family with graphics and presentation support, and one with compute support
    let queue_props = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    let mut compute_queue_family_index = std::u32::MAX;
    let mut graphics_queue_family_index = std::u32::MAX;
    for (i, queue) in queue_props.iter().enumerate() {
        if queue.queue_flags.contains(vk::QueueFlags::COMPUTE) {
            compute_queue_family_index = i as u32;
        }
        let supports_present = unsafe { surface_ext.get_physical_device_surface_support(physical_device, i as u32, surface) };
        if queue.queue_flags.contains(vk::QueueFlags::GRAPHICS) && supports_present {
            graphics_queue_family_index = i as u32;
        }
    }

    assert!(compute_queue_family_index != std::u32::MAX, "No compute queue family found!");
    assert!(graphics_queue_family_index != std::u32::MAX, "No graphics queue family found!");

    let priorities = [1.0];

    //Device queues must be specified when creating the device
    let mut queue_infos = vec![vk::DeviceQueueCreateInfo::builder()
                        .queue_family_index(compute_queue_family_index)
                        .queue_priorities(&priorities)
                        .build()];

    if compute_queue_family_index != graphics_queue_family_index {
        queue_infos.push(vk::DeviceQueueCreateInfo::builder()
                            .queue_family_index(graphics_queue_family_index)
                            .queue_priorities(&priorities)
                            .build());
    }

    //Specify that we want to use the swapchain extension on this device (necessary for presenting images to the screen)
    let device_extensions = [Swapchain::name().as_ptr()];
    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&device_extensions);
    let device = unsafe { instance.create_device(physical_device, &device_create_info, None).unwrap() };

    //Create memory allocator using the newly created device
    let mut allocator = {
            let create_info = vk_mem::AllocatorCreateInfo {
                physical_device,
                device: device.clone(),
                instance: instance.clone(),
                ..Default::default()
            };
            vk_mem::Allocator::new(&create_info).unwrap()
    };

    //Get handles to the compute and graphics queues
    //These may be identical if they come from the same queue family
    let compute_queue = unsafe { device.get_device_queue(compute_queue_family_index, 0) };
    let graphics_queue = unsafe { device.get_device_queue(graphics_queue_family_index, 0) };

    //Command buffers are allocated from command pools, and are unique to each queue family
    let compute_command_pool = {
        let pool_create = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(compute_queue_family_index);
        unsafe { device.create_command_pool(&pool_create, None).unwrap() }
    };

    //If the graphics queue family is separate from the compute queue family, create a separate command pool
    //just for graphics
    let graphics_command_pool = {
        if compute_queue_family_index == graphics_queue_family_index {
            compute_command_pool
        } else {
            let pool_create = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(graphics_queue_family_index);
            unsafe { device.create_command_pool(&pool_create, None).unwrap() }
        }
    };

    //Create a descriptor pool for allocating descriptor sets from
    //Only one set in total is needed for this app, and is only used for a storage buffer
    let desc_pool = {
        let desc_pool_size = [vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .build()];

        let create_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&desc_pool_size)
            .max_sets(1)
            .build();

        unsafe { device.create_descriptor_pool(&create_info, None).unwrap() }
    };

    //Create swapchain
    let swapchain_ext = Swapchain::new(&instance, &device);

    for formats in surface_formats.iter() {
        println!("{:?}", formats);
    }

    let swapchain = {
        let create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface)
            .min_image_count(2)
            .image_format(surface_formats[0].format)           //This method picks the first available format and color space
            .image_color_space(surface_formats[0].color_space) //For consistency, your app might search for a specific format/color space
            .image_extent(vk::Extent2D::builder().width(WINDOW_SIZE_X).height(WINDOW_SIZE_Y).build())
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO) //FIFO is guaranteed to be available
            .clipped(true);
        unsafe { swapchain_ext.create_swapchain(&create_info, None).unwrap() }
    };

    //Render passes describe the format of attachments that will be rendered to
    //This app will render into an image provided by the swapchain
    let render_pass = {

        //An attachment description describes the layout of the rendering attachment
        let attachment = [vk::AttachmentDescription::builder()
            .format(surface_formats[0].format) //Use the same format as the swapchain images
            .samples(vk::SampleCountFlags::TYPE_1) //No multisampling
            .load_op(vk::AttachmentLoadOp::CLEAR) //Clear this image when the render pass begins (clear value is specified later)
            .store_op(vk::AttachmentStoreOp::STORE) //Store this image at the end of rendering to present
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE) //No depth/stencil is used, so these can be dont care
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED) //This app doesn't read from the attachment, so this specifies the data is unknown
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR) //This layout is what the image will be moved to once the render pass ends
            .build()];

        //Attachment references describe the layout that each attachment should be in when the subpass begins
        let attach_refs = [vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build()];

        //Each renderpass is a collection of subpasses. This app only uses one pass to render
        let subpasses = [vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&attach_refs)
            .build()];

        //Subpass dependencies specify memory dependencies that must happen during subpass transitions
        //Image layout transitions normally automatically occur before the subpass begins using the layouts in
        //AttachmentDescription's and AttachmentReference's
        //For presentation, however, the swapchain image is usually not acquired yet, so this dependency
        //moves the layout transition to COLOR_ATTACHMENT to before the actual COLOR_ATTACHMENT_OUTPUT actually occurs
        let present_dependency = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT) 
            .src_access_mask(vk::AccessFlags::empty()) //Nothing needs to be waited on for the image to transition
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT) //Writes occur in the COLOR_ATTACHMENT_OUTPUT stage
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE) //Must transition image before writing to it
            .build();

        //Another dependency is inserted to ensure the compute shader has actually finished writing to the shared buffer
        let compute_dependency = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)         //All shader writes in the compute shader step must finish first
            .dst_stage_mask(vk::PipelineStageFlags::VERTEX_SHADER)  //Before any shader reads in the vertex shader occur
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .build();

        let dependencies = [present_dependency, compute_dependency];

        //Build the render pass
        let create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachment)
            .subpasses(&subpasses)
            .dependencies(&dependencies)
            .build();
        unsafe { device.create_render_pass(&create_info, None).unwrap() }
    };

    //Get handles to the actual swapchain images
    let swapchain_images = unsafe { swapchain_ext.get_swapchain_images(swapchain).unwrap() };

    //Create image views and framebuffers
    let mut swapchain_image_views = Vec::new();
    let mut framebuffers = Vec::new();

    for (i, image) in swapchain_images.iter().enumerate() {
        //Image views describe access on a subset of an image resource (i.e. a few mipmap layers)
        //As the swapchain images should not use mipmapping and aren't array images, the image view should cover the entire image
        let create_info = vk::ImageViewCreateInfo::builder()
            .image(*image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(surface_formats[0].format)
            .components(vk::ComponentMapping::builder().r(vk::ComponentSwizzle::IDENTITY).g(vk::ComponentSwizzle::IDENTITY).b(vk::ComponentSwizzle::IDENTITY).a(vk::ComponentSwizzle::IDENTITY).build())
            .subresource_range(vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1)
                                .build());
        let iv = unsafe { device.create_image_view(&create_info, None).unwrap() };
        swapchain_image_views.push(iv);

        //Framebuffers specify a particular image view to use as an attachment. These will be used with the render pass created above
        let attachments = [swapchain_image_views[i]];

        let create_info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(WINDOW_SIZE_X)
            .height(WINDOW_SIZE_Y)
            .layers(1)
            .build();

        let fb = unsafe { device.create_framebuffer(&create_info, None).unwrap() };

        framebuffers.push(fb);
    }

    //Descriptor set layouts are used to define how resources are access by a shader
    //As the vertex and compute shaders expect a storage buffer at binding 0, a layout for this should be created

    let desc_set_layout = {
        let binding = [vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX)
            .build()];
        let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&binding)
            .build();
        unsafe { device.create_descriptor_set_layout(&create_info, None).unwrap() }
    };

    let desc_set_layouts = [desc_set_layout];

    //A descriptor set is used to actually bind a resource to a particular binding point
    let desc_set = {
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(desc_pool)
            .set_layouts(&desc_set_layouts)
            .build();
        let sets = unsafe { device.allocate_descriptor_sets(&alloc_info).unwrap() };
        sets[0]
    };

    //A pipeline layout is a collection of all of the descriptor set layouts and push constants that will be used in a single pipeline
    let pipeline_layout = {
        let create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&desc_set_layouts)
            .build();
        unsafe { device.create_pipeline_layout(&create_info, None).unwrap() }
    };

    //Create storage buffer and allocate memory for it (handled by vk-mem)
    let (storage_buffer, storage_allocation, _info) = {
        let buf_create_info = vk::BufferCreateInfo::builder()
            .size(BUFFER_SIZE)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER);
        let alloc_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        };

        allocator.create_buffer(&buf_create_info, &alloc_create_info).unwrap()
    };

    //Bind the buffer to the descriptor set
    {
        let buf_info = [vk::DescriptorBufferInfo::builder()
            .buffer(storage_buffer)
            .offset(0)
            .range(BUFFER_SIZE)
            .build()];
        let write = [vk::WriteDescriptorSet::builder()
            .dst_set(desc_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buf_info)
            .build()];

        unsafe { device.update_descriptor_sets(&write, &[]) };
    }

    //Create a pipeline around the compute shader
    let compute_pipeline = {
        let shader_module = {
            //SPIR-V modules are specified with u32 arrays instead of u8 arrays, so do the proper conversion
            let shader_spv = include_bytes!("../comp.spv");
            assert!(shader_spv.len() % 4 == 0, "Invalid SPV format");

            let mut spv_code = vec![0; shader_spv.len() / 4];
            NativeEndian::read_u32_into(shader_spv, spv_code.as_mut_slice());

            let create_info = vk::ShaderModuleCreateInfo::builder()
                .code(spv_code.as_slice())
                .build();
            unsafe { device.create_shader_module(&create_info, None).unwrap() }
        };

        let entrypoint = CString::new("main").unwrap();
        let shader_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entrypoint)
            .build();
        let create_info = [vk::ComputePipelineCreateInfo::builder()
            .stage(shader_stage)
            .layout(pipeline_layout)
            .build()];
        let pipelines = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &create_info, None).unwrap() };
        unsafe { device.destroy_shader_module(shader_module, None) };
        pipelines[0]
    };

    //Create a graphics pipeline around the vertex and fragment shaders
    let graphics_pipeline = {
        let f_spv = include_bytes!("../frag.spv");
        let v_spv = include_bytes!("../vert.spv");

        let mut f_code = vec![0; f_spv.len() / 4];
        let mut v_code = vec![0; v_spv.len() / 4];

        NativeEndian::read_u32_into(v_spv, v_code.as_mut_slice());
        NativeEndian::read_u32_into(f_spv, f_code.as_mut_slice());

        let create_info = vk::ShaderModuleCreateInfo::builder()
            .code(f_code.as_slice())
            .build();
        let f_mod = unsafe { device.create_shader_module(&create_info, None).unwrap() };

        let create_info = vk::ShaderModuleCreateInfo::builder()
            .code(v_code.as_slice())
            .build();
        let v_mod = unsafe { device.create_shader_module(&create_info, None).unwrap() };

        let entrypoint = CString::new("main").unwrap();
        let v_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(v_mod)
            .name(&entrypoint)
            .build();
        let f_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(f_mod)
            .name(&entrypoint)
            .build();

        //Points are read from a storage buffer, so no vertex input is necessary
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder().build();

        //Verticies will be drawn as points
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::POINT_LIST)
            .primitive_restart_enable(false)
            .build();

        //Standard fill for rasterization
        let raster_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .line_width(1.0)
            .build();

        //Viewport and scissor cover the entire screen
        let viewport = [vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(WINDOW_SIZE_X as f32)
            .height(WINDOW_SIZE_Y as f32)
            .min_depth(0.0)
            .max_depth(1.0)
            .build()];
        let scissor = [vk::Rect2D::builder()
            .offset(vk::Offset2D::builder().x(0).y(0).build())
            .extent(vk::Extent2D::builder().width(WINDOW_SIZE_X).height(WINDOW_SIZE_Y).build())
            .build()];

        let view_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewport)
            .scissors(&scissor)
            .build();

        //No multisampling
        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false)
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false)
            .build();

        //No blending
        let blend_attachment = [vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::all())
            .build()];

        let blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(&blend_attachment)
            .build();

        let stages = [v_stage, f_stage];

        let create_info = [vk::GraphicsPipelineCreateInfo::builder()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&view_state)
            .rasterization_state(&raster_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&blend_state)
            .render_pass(render_pass)
            .subpass(0)
            .layout(pipeline_layout)
            .build()];
        let pipelines = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &create_info, None).unwrap() };
        unsafe {
            device.destroy_shader_module(v_mod, None);
            device.destroy_shader_module(f_mod, None);
        }
        pipelines[0]
    };

    //Allocate command buffers for submitting compute and graphics work
    let compute_command_buffer = {
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(compute_command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let buffers = unsafe { device.allocate_command_buffers(&alloc_info).unwrap() };
        buffers[0]
    };

    let graphics_command_buffer = {
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(graphics_command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let buffers = unsafe { device.allocate_command_buffers(&alloc_info).unwrap() };
        buffers[0]
    };

    //Create semaphores for syncing between when the swapchain image is ready for writing, the render is finished, and the compute shader invocation is finished
    let (image_ready_semaphore, render_finished_semaphore, compute_finished_semaphore) = {
        let create_info = vk::SemaphoreCreateInfo::builder().build();
        unsafe { (device.create_semaphore(&create_info, None).unwrap(), device.create_semaphore(&create_info, None).unwrap(), device.create_semaphore(&create_info, None).unwrap()) }
    };

    let mut events = sdl_context.event_pump().unwrap();
    'running: loop {
        for event in events.poll_iter() {
            match event {
                Event::Quit{..} => {
                    break 'running
                },
                _ => {}
            }
        }

        unsafe { device.device_wait_idle().unwrap() };

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .build();

        let sets = [desc_set];

        //A compute shader run consists of binding the pipeline, binding the descriptor set to access the buffer, then dispatching the shader
        unsafe {
            device.begin_command_buffer(compute_command_buffer, &begin_info).unwrap();
            device.cmd_bind_pipeline(compute_command_buffer, vk::PipelineBindPoint::COMPUTE, compute_pipeline);
            device.cmd_bind_descriptor_sets(compute_command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline_layout, 0, &sets, &[]);
            device.cmd_dispatch(compute_command_buffer, 1, 1, 1);
            device.end_command_buffer(compute_command_buffer).unwrap();
        }

        let buffers = [compute_command_buffer];
        let signal_semaphores = [compute_finished_semaphore];
        let submit = [vk::SubmitInfo::builder()
            .command_buffers(&buffers)
            .signal_semaphores(&signal_semaphores)
            .build()];

        unsafe { device.queue_submit(compute_queue, &submit, vk::Fence::null()).unwrap() };

        let (fb_idx, _) = unsafe { swapchain_ext.acquire_next_image(swapchain, std::u64::MAX, image_ready_semaphore, vk::Fence::null()).unwrap() };

        //The graphics task consists of beginning the renderpass (which will handle clearing the image), binding the graphics pipeline,
        //Binding the descriptor set to access the shared buffer, the drawing the points
        unsafe { device.begin_command_buffer(graphics_command_buffer, &begin_info).unwrap() };
        let clear_value = vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0 ]};
        let clear_value = [vk::ClearValue {color: clear_value}];
        let rp_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(render_pass)
            .framebuffer(framebuffers[fb_idx as usize])
            .render_area(vk::Rect2D::builder().offset(vk::Offset2D::builder().x(0).y(0).build()).extent(vk::Extent2D::builder().width(WINDOW_SIZE_X).height(WINDOW_SIZE_Y).build()).build())
            .clear_values(&clear_value)
            .build();

        unsafe {
            device.cmd_begin_render_pass(graphics_command_buffer, &rp_begin_info, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(graphics_command_buffer, vk::PipelineBindPoint::GRAPHICS, graphics_pipeline);
            device.cmd_bind_descriptor_sets(graphics_command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline_layout, 0, &sets, &[]);
            device.cmd_draw(graphics_command_buffer, NUM_PARTICLES, 1, 0, 0);
            device.cmd_end_render_pass(graphics_command_buffer);
            device.end_command_buffer(graphics_command_buffer).unwrap();
        }

        let wait_semaphores = [image_ready_semaphore, compute_finished_semaphore];
        let dst_stage_mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::PipelineStageFlags::VERTEX_SHADER];
        let cmd_buffers = [graphics_command_buffer];
        let signal_semaphores = [render_finished_semaphore];

        let submit  = [vk::SubmitInfo::builder()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&dst_stage_mask)
            .command_buffers(&cmd_buffers)
            .signal_semaphores(&signal_semaphores)
            .build()];
        unsafe { device.queue_submit(graphics_queue, &submit, vk::Fence::null()).unwrap() };

        let wait_semaphores = [render_finished_semaphore];
        let swapchains = [swapchain];
        let image_indices = [fb_idx];

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices)
            .build();
        unsafe { swapchain_ext.queue_present(graphics_queue, &present_info).unwrap() };
    }
    unsafe { 
        device.device_wait_idle().unwrap();

        //Cleanup
        device.destroy_semaphore(image_ready_semaphore, None);
        device.destroy_semaphore(render_finished_semaphore, None);
        device.destroy_semaphore(compute_finished_semaphore, None);
        allocator.destroy_buffer(storage_buffer, &storage_allocation).unwrap();
        drop(allocator);

        for iv in swapchain_image_views.iter() {
            device.destroy_image_view(*iv, None);
        }
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_render_pass(render_pass, None);
        device.destroy_pipeline(compute_pipeline, None);
        device.destroy_pipeline(graphics_pipeline, None);
        device.destroy_descriptor_pool(desc_pool, None);
        device.destroy_descriptor_set_layout(desc_set_layout, None);
        for fb in framebuffers.iter() {
            device.destroy_framebuffer(*fb, None);
        }
        if compute_command_pool != graphics_command_pool {
            device.destroy_command_pool(compute_command_pool, None);
            device.destroy_command_pool(graphics_command_pool, None);
        } else {
            device.destroy_command_pool(compute_command_pool, None);
        }
        
        swapchain_ext.destroy_swapchain(swapchain, None);
        device.destroy_device(None);

        debug_report_loader.destroy_debug_report_callback(debug_call_back, None);
        instance.destroy_instance(None);
    }
}
