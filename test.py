import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from io import BytesIO
import cv2

# Set page config
st.set_page_config(page_title="Pixel Art to Realistic Image Converter", layout="wide")

# App title
st.title("🖼 Pixel Art to Realistic Image Converter")
st.markdown("Transform *pixel art* back into *photorealistic images* using advanced AI techniques!")

# Image uploader
uploaded_file = st.file_uploader("Upload your pixel art", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load image using PIL
        pixel_image = Image.open(uploaded_file)
        
        # Convert to RGB if needed
        if pixel_image.mode in ('RGBA', 'LA', 'P'):
            pixel_image = pixel_image.convert('RGB')
        
        # Show original pixel art
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Pixel Art")
            st.image(pixel_image, caption=f"Size: {pixel_image.width}×{pixel_image.height}", use_container_width=True)
        
        # Get original dimensions
        orig_width, orig_height = pixel_image.size
        
        # Enhancement parameters
        st.subheader("🎛 Realistic Image Generation Controls")
        
        # Output Size Configuration
        with st.expander("📐 Output Size Configuration", expanded=True):
            size_mode = st.radio(
                "Size Control Method",
                ["Upscale Factor", "Custom Dimensions", "Preset Sizes"],
                help="Choose how to control the output image size"
            )
            
            col_size1, col_size2 = st.columns(2)
            
            if size_mode == "Upscale Factor":
                with col_size1:
                    upscale_factor = st.selectbox(
                        "Upscale Factor",
                        [2, 4, 6, 8, 10, 12, 16, 20],
                        index=3,  # Default to 8x
                        help="Multiply original dimensions by this factor"
                    )
                    target_width = orig_width * upscale_factor
                    target_height = orig_height * upscale_factor
                
                with col_size2:
                    st.info(f"*Output Size:* {target_width} × {target_height} pixels")
                    aspect_ratio = orig_width / orig_height
                    st.caption(f"Original aspect ratio: {aspect_ratio:.2f}:1")
            
            elif size_mode == "Custom Dimensions":
                with col_size1:
                    target_width = st.number_input(
                        "Output Width (pixels)",
                        min_value=50,
                        max_value=8000,
                        value=orig_width * 4,
                        step=50,
                        help="Specify exact width in pixels"
                    )
                    
                    target_height = st.number_input(
                        "Output Height (pixels)",
                        min_value=50,
                        max_value=8000,
                        value=orig_height * 4,
                        step=50,
                        help="Specify exact height in pixels"
                    )
                
                with col_size2:
                    # Calculate upscale factors for each dimension
                    width_scale = target_width / orig_width
                    height_scale = target_height / orig_height
                    
                    st.info(f"*Width Scale:* {width_scale:.1f}x")
                    st.info(f"*Height Scale:* {height_scale:.1f}x")
                    
                    # Aspect ratio controls
                    maintain_aspect = st.checkbox(
                        "🔒 Maintain Aspect Ratio",
                        value=False,
                        help="Keep original proportions (may override height)"
                    )
                    
                    if maintain_aspect:
                        # Recalculate height based on width to maintain aspect ratio
                        original_ratio = orig_width / orig_height
                        target_height = int(target_width / original_ratio)
                        st.caption(f"Height adjusted to {target_height} to maintain aspect ratio")
                
                # Set upscale_factor for compatibility with existing code
                upscale_factor = max(width_scale, height_scale)
            
            else:  # Preset Sizes
                with col_size1:
                    preset_options = {
                        "Instagram Square (1080×1080)": (1080, 1080),
                        "Instagram Portrait (1080×1350)": (1080, 1350),
                        "Facebook Cover (1200×630)": (1200, 630),
                        "Twitter Header (1500×500)": (1500, 500),
                        "HD (1920×1080)": (1920, 1080),
                        "4K (3840×2160)": (3840, 2160),
                        "Print 4×6 (1800×1200)": (1800, 1200),
                        "Print 8×10 (3000×2400)": (3000, 2400),
                        "A4 Portrait (2480×3508)": (2480, 3508),
                        "A4 Landscape (3508×2480)": (3508, 2480)
                    }
                    
                    selected_preset = st.selectbox(
                        "Choose Preset Size",
                        list(preset_options.keys()),
                        help="Select from common social media and print sizes"
                    )
                    
                    target_width, target_height = preset_options[selected_preset]
                
                with col_size2:
                    st.info(f"*Selected Size:* {target_width} × {target_height}")
                    
                    # Option to fit or fill
                    fit_mode = st.radio(
                        "Scaling Mode",
                        ["Fit (preserve aspect)", "Fill (stretch to fit)", "Crop to fit"],
                        help="How to handle aspect ratio differences"
                    )
                    
                    if fit_mode == "Fit (preserve aspect)":
                        # Calculate the scale to fit within preset dimensions
                        scale_w = target_width / orig_width
                        scale_h = target_height / orig_height
                        scale = min(scale_w, scale_h)
                        
                        # Adjust dimensions to maintain aspect ratio
                        fitted_width = int(orig_width * scale)
                        fitted_height = int(orig_height * scale)
                        
                        st.caption(f"Actual output: {fitted_width} × {fitted_height}")
                        st.caption("(Image will be centered with padding if needed)")
                        
                        target_width, target_height = fitted_width, fitted_height
                
                # Set upscale_factor for compatibility
                upscale_factor = max(target_width / orig_width, target_height / orig_height)
        
        # Quality enhancement options
        with st.expander("🚀 AI Enhancement Options", expanded=True):
            col_q1, col_q2 = st.columns(2)
            
            with col_q1:
                st.markdown("🔧 Processing Options:")
                
                enhancement_strength = st.slider(
                    "Enhancement Strength",
                    min_value=0.1,
                    max_value=3.0,
                    value=1.5,
                    step=0.1,
                    help="Higher values = more dramatic enhancement"
                )
                
                noise_reduction = st.checkbox("🔇 Noise Reduction", value=True, help="Reduces pixelation artifacts")
                edge_enhancement = st.checkbox("📐 Edge Enhancement", value=True, help="Sharpens important edges")
            
            with col_q2:
                st.markdown("🎨 Realism Options:")
                
                realism_mode = st.selectbox(
                    "Realism Mode",
                    ["Photorealistic", "Artistic", "Smooth", "Sharp Details"],
                    help="Different approaches to making images look realistic"
                )
                
                color_enhancement = st.checkbox("🌈 Color Enhancement", value=True, help="Improves color richness and depth")
                texture_synthesis = st.checkbox("🎯 Texture Synthesis", value=True, help="Adds realistic textures")
                lighting_enhancement = st.checkbox("💡 Lighting Enhancement", value=True, help="Improves lighting and shadows")
                
                preserve_style = st.checkbox("🎪 Preserve Original Style", value=False, help="Keeps some pixel art characteristics")
        
        # Advanced options
        with st.expander("⚙ Advanced Settings"):
            col_a1, col_a2 = st.columns(2)
            
            with col_a1:
                # Multi-step processing
                multi_step = st.checkbox("🔄 Multi-Step Processing", value=True, help="Uses multiple enhancement passes")
                iterations = st.slider("Enhancement Iterations", 1, 5, 3) if multi_step else 1
                
                # Smoothing options
                smoothing_method = st.selectbox(
                    "Smoothing Algorithm",
                    ["Gaussian", "Bilateral", "Non-Local Means", "Edge-Preserving"],
                    index=2,
                    help="Different methods for smoothing pixelated edges"
                )
            
            with col_a2:
                # Detail enhancement
                detail_boost = st.slider("Detail Boost", 0.0, 2.0, 1.0, 0.1)
                contrast_boost = st.slider("Contrast Boost", 0.5, 2.0, 1.2, 0.1)
                saturation_boost = st.slider("Saturation Boost", 0.5, 2.0, 1.3, 0.1)
                
                # Output format options
                output_format = st.selectbox("Output Format", ["PNG", "JPEG"], help="Choose output file format")
        
        # Target dimensions are now set above based on size mode selection
        
        # Show processing info with more details
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info(f"🎯 *Input:* {orig_width}×{orig_height} pixels")
        with col_info2:
            st.info(f"🎯 *Output:* {target_width}×{target_height} pixels")
        
        # Show scaling information
        if size_mode == "Custom Dimensions":
            width_scale = target_width / orig_width
            height_scale = target_height / orig_height
            if abs(width_scale - height_scale) > 0.1:
                st.warning(f"⚠ Non-uniform scaling detected: Width {width_scale:.1f}x, Height {height_scale:.1f}x. Image may appear stretched.")
        elif size_mode == "Preset Sizes":
            st.info(f"📱 Using preset: {selected_preset} with {fit_mode.lower()}")
        else:
            st.info(f"📈 Upscaling by {upscale_factor}x factor")
        
        def safe_image_resize(img, target_size):
            """Safely resize image with compatibility for different PIL versions"""
            try:
                # Try new Pillow syntax first (Pillow >= 10.0.0)
                return img.resize(target_size, Image.Resampling.LANCZOS)
            except AttributeError:
                try:
                    # Try older Pillow syntax (Pillow 9.x)
                    return img.resize(target_size, Image.LANCZOS)
                except AttributeError:
                    # Fallback to even older syntax
                    return img.resize(target_size, Image.ANTIALIAS)
        
        def apply_advanced_smoothing(img_array, method="Gaussian", strength=1.0):
            """Apply advanced smoothing algorithms with better error handling"""
            try:
                if method == "Gaussian":
                    # Gaussian blur with edge preservation
                    kernel_size = max(3, int(5 * strength))
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    smoothed = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), strength)
                    
                elif method == "Bilateral":
                    # Bilateral filter - preserves edges while smoothing
                    d = max(5, min(15, int(9 * strength)))  # Limit d parameter
                    sigma_color = min(150, 75 * strength)
                    sigma_space = min(150, 75 * strength)
                    smoothed = cv2.bilateralFilter(img_array, d, sigma_color, sigma_space)
                    
                elif method == "Non-Local Means":
                    # Non-local means denoising - best for pixel art
                    h = min(30, 10 * strength)  # Limit h parameter
                    template_window_size = 7
                    search_window_size = 21
                    smoothed = cv2.fastNlMeansDenoisingColored(img_array, None, h, h, template_window_size, search_window_size)
                    
                else:  # Edge-Preserving
                    # Edge-preserving filter
                    flags = 2  # RECURS_FILTER
                    sigma_s = min(200, 50 * strength)
                    sigma_r = min(1.0, 0.4 * strength)
                    smoothed = cv2.edgePreservingFilter(img_array, flags=flags, sigma_s=sigma_s, sigma_r=sigma_r)
                    
                return smoothed
                
            except Exception as e:
                # Fallback to simple Gaussian blur
                st.warning(f"Advanced smoothing failed ({method}), using Gaussian blur")
                kernel_size = max(3, int(3 * strength))
                if kernel_size % 2 == 0:
                    kernel_size += 1
                return cv2.GaussianBlur(img_array, (kernel_size, kernel_size), max(0.5, strength))
        
        def enhance_edges_advanced(img_array, strength=1.0):
            """Advanced edge enhancement using unsharp masking"""
            try:
                # Convert to float for processing
                img_float = img_array.astype(np.float32) / 255.0
                
                # Create Gaussian blur
                blur_strength = max(0.5, min(5.0, 2.0 * strength))
                blurred = cv2.GaussianBlur(img_float, (0, 0), blur_strength)
                
                # Unsharp mask
                unsharp_mask = img_float - blurred
                enhanced = img_float + min(2.0, strength) * unsharp_mask
                
                # Clip values and convert back
                enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
                return enhanced
            except Exception as e:
                st.warning(f"Edge enhancement failed, skipping")
                return img_array
        
        def synthesize_textures(img_array, strength=1.0):
            """Add subtle texture synthesis for realism"""
            try:
                height, width = img_array.shape[:2]
                
                # Create subtle noise pattern with limited strength
                noise_strength = min(10, 5 * strength)
                noise = np.random.normal(0, noise_strength, (height, width, 3))
                
                # Apply noise based on image brightness (darker areas get more texture)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                texture_mask = (255 - gray) / 255.0  # Invert brightness
                texture_mask = np.stack([texture_mask] * 3, axis=2)
                
                # Apply textured noise with limited intensity
                noise_intensity = min(0.5, 0.3 * strength)
                textured = img_array.astype(np.float32) + noise * texture_mask * noise_intensity
                textured = np.clip(textured, 0, 255).astype(np.uint8)
                
                return textured
            except Exception as e:
                st.warning(f"Texture synthesis failed, skipping")
                return img_array
        
        def enhance_lighting_depth(img_array, strength=1.0):
            """Enhance lighting and add depth perception"""
            try:
                # Convert to LAB color space for better lighting control
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Enhance L channel (lightness) with adaptive histogram equalization
                clip_limit = min(5.0, 2.0 * strength)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                l_enhanced = clahe.apply(l)
                
                # Merge back
                lab_enhanced = cv2.merge([l_enhanced, a, b])
                rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
                
                return rgb_enhanced
            except Exception as e:
                st.warning(f"Lighting enhancement failed, skipping")
                return img_array
        
        def super_resolution_upscale(img, target_width, target_height, method="INTER_CUBIC"):
            """Advanced upscaling with multiple interpolation methods"""
            img_array = np.array(img)
            
            try:
                if method == "INTER_CUBIC":
                    upscaled = cv2.resize(img_array, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
                elif method == "INTER_LANCZOS4":
                    upscaled = cv2.resize(img_array, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
                else:  # Multi-step upscaling
                    # Progressive upscaling for better quality
                    current_img = img_array
                    current_w, current_h = img.width, img.height
                    
                    # Limit the number of upscaling steps to prevent memory issues
                    max_steps = 5
                    step_count = 0
                    
                    while (current_w < target_width or current_h < target_height) and step_count < max_steps:
                        # Double the size each step
                        next_w = min(current_w * 2, target_width)
                        next_h = min(current_h * 2, target_height)
                        
                        current_img = cv2.resize(current_img, (next_w, next_h), interpolation=cv2.INTER_CUBIC)
                        current_w, current_h = next_w, next_h
                        step_count += 1
                    
                    # Final resize to exact target dimensions if needed
                    if current_w != target_width or current_h != target_height:
                        current_img = cv2.resize(current_img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
                    
                    upscaled = current_img
                    
                return upscaled
                
            except Exception as e:
                # Fallback to basic cubic interpolation
                st.warning(f"Advanced upscaling failed, using basic method")
                return cv2.resize(img_array, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        
        def create_realistic_image(pixel_img, target_width, target_height, realism_mode, 
                                 enhancement_strength, noise_reduction, edge_enhancement,
                                 color_enhancement, texture_synthesis_flag, lighting_enhancement,
                                 preserve_style, smoothing_method, iterations, detail_boost,
                                 contrast_boost, saturation_boost, size_mode="Upscale Factor", fit_mode="Fill"):
            """Main function to convert pixel art to realistic image"""
            
            try:
                # Check for reasonable dimensions to prevent memory issues
                max_dimension = 8000
                if target_width > max_dimension or target_height > max_dimension:
                    st.warning(f"Target dimensions too large. Limiting to {max_dimension}x{max_dimension}")
                    scale = min(max_dimension / target_width, max_dimension / target_height)
                    target_width = int(target_width * scale)
                    target_height = int(target_height * scale)
                
                # Handle different fitting modes for preset sizes
                if size_mode == "Preset Sizes" and 'fit_mode' in locals():
                    if fit_mode == "Crop to fit":
                        # First upscale to larger dimension, then crop
                        orig_aspect = pixel_img.width / pixel_img.height
                        target_aspect = target_width / target_height
                        
                        if orig_aspect > target_aspect:
                            # Original is wider, scale by height
                            temp_height = target_height
                            temp_width = int(temp_height * orig_aspect)
                        else:
                            # Original is taller, scale by width
                            temp_width = target_width
                            temp_height = int(temp_width / orig_aspect)
                        
                        # Upscale to temporary dimensions
                        upscaled_array = super_resolution_upscale(pixel_img, temp_width, temp_height, "INTER_CUBIC")
                        upscaled_img = Image.fromarray(upscaled_array)
                        
                        # Crop to center
                        left = (temp_width - target_width) // 2
                        top = (temp_height - target_height) // 2
                        right = left + target_width
                        bottom = top + target_height
                        
                        upscaled_img = upscaled_img.crop((left, top, right, bottom))
                        upscaled_array = np.array(upscaled_img)
                        
                    elif fit_mode == "Fit (preserve aspect)":
                        # Scale to fit within dimensions, then pad if needed
                        scale_w = target_width / pixel_img.width
                        scale_h = target_height / pixel_img.height
                        scale = min(scale_w, scale_h)
                        
                        fitted_width = int(pixel_img.width * scale)
                        fitted_height = int(pixel_img.height * scale)
                        
                        # Upscale to fitted dimensions
                        upscaled_array = super_resolution_upscale(pixel_img, fitted_width, fitted_height, "INTER_CUBIC")
                        
                        # Create canvas and center the image
                        canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * 128  # Gray background
                        
                        y_offset = (target_height - fitted_height) // 2
                        x_offset = (target_width - fitted_width) // 2
                        
                        canvas[y_offset:y_offset+fitted_height, x_offset:x_offset+fitted_width] = upscaled_array
                        upscaled_array = canvas
                    else:
                        # Fill mode - stretch to fit
                        upscaled_array = super_resolution_upscale(pixel_img, target_width, target_height, "INTER_CUBIC")
                else:
                    # Standard upscaling
                    if realism_mode == "Sharp Details":
                        upscaled_array = super_resolution_upscale(pixel_img, target_width, target_height, "INTER_LANCZOS4")
                    else:
                        upscaled_array = super_resolution_upscale(pixel_img, target_width, target_height, "multi-step")
                
                # Step 2: Multiple enhancement iterations (limit iterations to prevent excessive processing)
                enhanced_array = upscaled_array.copy()
                actual_iterations = min(iterations, 3)  # Limit to 3 iterations max
                
                for iteration in range(actual_iterations):
                    # Noise reduction and smoothing
                    if noise_reduction and not preserve_style:
                        smoothing_strength = enhancement_strength * (0.8 if iteration == 0 else 0.4)
                        enhanced_array = apply_advanced_smoothing(enhanced_array, smoothing_method, smoothing_strength)
                    
                    # Edge enhancement
                    if edge_enhancement:
                        edge_strength = detail_boost * (1.0 if iteration == 0 else 0.6)
                        enhanced_array = enhance_edges_advanced(enhanced_array, edge_strength)
                    
                    # Texture synthesis
                    if texture_synthesis_flag and not preserve_style:
                        texture_strength = enhancement_strength * 0.5
                        enhanced_array = synthesize_textures(enhanced_array, texture_strength)
                
                # Step 3: Lighting and depth enhancement
                if lighting_enhancement:
                    enhanced_array = enhance_lighting_depth(enhanced_array, enhancement_strength)
                
                # Step 4: Color enhancement
                if color_enhancement:
                    enhanced_img = Image.fromarray(enhanced_array)
                    
                    # Enhance contrast
                    if contrast_boost != 1.0:
                        enhancer = ImageEnhance.Contrast(enhanced_img)
                        enhanced_img = enhancer.enhance(max(0.1, min(3.0, contrast_boost)))
                    
                    # Enhance color saturation
                    if saturation_boost != 1.0:
                        enhancer = ImageEnhance.Color(enhanced_img)
                        enhanced_img = enhancer.enhance(max(0.1, min(3.0, saturation_boost)))
                    
                    enhanced_array = np.array(enhanced_img)
                
                # Step 5: Final realism adjustments based on mode
                if realism_mode == "Photorealistic":
                    # Apply subtle film grain for photorealism
                    grain = np.random.normal(0, 2, enhanced_array.shape)
                    enhanced_array = enhanced_array.astype(np.float32) + grain * 0.3
                    enhanced_array = np.clip(enhanced_array, 0, 255).astype(np.uint8)
                    
                elif realism_mode == "Artistic":
                    # Slight stylization while maintaining realism
                    enhanced_array = cv2.bilateralFilter(enhanced_array, 9, 80, 80)
                    
                elif realism_mode == "Smooth":
                    # Extra smoothing for clean look
                    enhanced_array = cv2.GaussianBlur(enhanced_array, (3, 3), 1.0)
                
                # Convert back to PIL Image
                realistic_image = Image.fromarray(enhanced_array)
                
                return realistic_image
                
            except Exception as e:
                st.error(f"Error during image processing: {str(e)}")
                # Return a basic upscaled version as fallback
                return safe_image_resize(pixel_img, (target_width, target_height))
        
        # Generate the realistic image
        if st.button("🚀 Generate Realistic Image", type="primary"):
            with st.spinner("🔄 Converting pixel art to realistic image... This may take a moment."):
                try:
                    realistic_image = create_realistic_image(
                        pixel_image, target_width, target_height, realism_mode,
                        enhancement_strength, noise_reduction, edge_enhancement,
                        color_enhancement, texture_synthesis, lighting_enhancement,
                        preserve_style, smoothing_method, iterations, detail_boost,
                        contrast_boost, saturation_boost, size_mode, 
                        fit_mode if size_mode == "Preset Sizes" else "Fill"
                    )
                    
                    # Store the result in session state
                    st.session_state.realistic_image = realistic_image
                    st.session_state.target_width = target_width
                    st.session_state.target_height = target_height
                    st.session_state.upscale_factor = upscale_factor
                    st.session_state.output_format = output_format
                    
                except Exception as e:
                    st.error(f"Failed to process image: {str(e)}")
                    st.session_state.realistic_image = safe_image_resize(pixel_image, (target_width, target_height))
        
        # Show realistic image if it exists in session state
        if hasattr(st.session_state, 'realistic_image') and st.session_state.realistic_image:
            with col2:
                st.subheader("Generated Realistic Image")
                st.image(
                    st.session_state.realistic_image, 
                    caption=f"Realistic: {st.session_state.target_width}×{st.session_state.target_height} ({st.session_state.upscale_factor}x upscaled)",
                    use_container_width=True
                )
            
            # Show processing details
            if st.checkbox("📊 Show Processing Details", value=False):
                st.subheader("Processing Pipeline")
                
                col_p1, col_p2, col_p3 = st.columns(3)
                
                with col_p1:
                    st.markdown("🔧 Applied Enhancements:")
                    enhancements = []
                    if noise_reduction: enhancements.append("✅ Noise Reduction")
                    if edge_enhancement: enhancements.append("✅ Edge Enhancement")
                    if color_enhancement: enhancements.append("✅ Color Enhancement")
                    if texture_synthesis: enhancements.append("✅ Texture Synthesis")
                    if lighting_enhancement: enhancements.append("✅ Lighting Enhancement")
                    
                    for enhancement in enhancements:
                        st.text(enhancement)
                
                with col_p2:
                    st.markdown("⚙ Settings Used:")
                    st.text(f"Realism Mode: {realism_mode}")
                    st.text(f"Smoothing: {smoothing_method}")
                    st.text(f"Iterations: {iterations}")
                    st.text(f"Enhancement Strength: {enhancement_strength}")
                
                with col_p3:
                    st.markdown("📈 Quality Metrics:")
                    st.text(f"Upscale Factor: {st.session_state.upscale_factor}x")
                    st.text(f"Detail Boost: {detail_boost}")
                    st.text(f"Contrast Boost: {contrast_boost}")
                    st.text(f"Saturation Boost: {saturation_boost}")
            
            # Download options
            st.subheader("💾 Download Realistic Image")
            
            col_d1, col_d2, col_d3 = st.columns(3)
            
            with col_d1:
                # Download standard quality
                try:
                    buf_standard = BytesIO()
                    quality = 95 if st.session_state.output_format == "JPEG" else None
                    st.session_state.realistic_image.save(buf_standard, format=st.session_state.output_format, quality=quality)
                    byte_standard = buf_standard.getvalue()
                    
                    st.download_button(
                        label=f"📱 Standard Quality ({st.session_state.output_format})",
                        data=byte_standard,
                        file_name=f"realistic_image_standard.{st.session_state.output_format.lower()}",
                        mime=f"image/{st.session_state.output_format.lower()}",
                        help="High quality for general use"
                    )
                except Exception as e:
                    st.error(f"Error preparing download: {str(e)}")
            
            with col_d2:
                # Download high quality PNG
                try:
                    buf_hq = BytesIO()
                    st.session_state.realistic_image.save(buf_hq, format="PNG", optimize=False)
                    byte_hq = buf_hq.getvalue()
                    
                    st.download_button(
                        label="🖼 High Quality (PNG)",
                        data=byte_hq,
                        file_name="realistic_image_high_quality.png",
                        mime="image/png",
                        help="Maximum quality PNG"
                    )
                except Exception as e:
                    st.error(f"Error preparing PNG download: {str(e)}")
            
            with col_d3:
                # Download print quality (if large enough)
                if st.session_state.target_width >= 1000 or st.session_state.target_height >= 1000:
                    try:
                        buf_print = BytesIO()
                        st.session_state.realistic_image.save(buf_print, format="PNG", optimize=True)
                        byte_print = buf_print.getvalue()
                        
                        st.download_button(
                            label="🖨 Print Quality",
                            data=byte_print,
                            file_name="realistic_image_print_quality.png",
                            mime="image/png",
                            help="Optimized for printing"
                        )
                    except Exception as e:
                        st.error(f"Error preparing print quality download: {str(e)}")
                else:
                    st.info("💡 Increase upscale factor for print quality")
            
            # Show enhanced stats
            st.subheader("📊 Conversion Statistics")
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            
            with col_s1:
                st.metric("Original Size", f"{orig_width}×{orig_height}")
            with col_s2:
                st.metric("Final Size", f"{st.session_state.target_width}×{st.session_state.target_height}")
            with col_s3:
                st.metric("Upscale Factor", f"{st.session_state.upscale_factor}x")
            with col_s4:
                size_increase = (st.session_state.target_width * st.session_state.target_height) / (orig_width * orig_height)
                st.metric("Pixel Increase", f"{size_increase:.1f}x")
    
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        st.info("Please try uploading a different image or check if the image file is corrupted.")

else:
    st.info("⬆ Please upload a pixel art image to start the conversion to realistic image!")
    
    # Show features
    st.subheader("🚀 Advanced AI Features:")
    
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        st.markdown("""
        *🎨 AI Enhancement:*
        - Multi-step super-resolution upscaling
        - Advanced noise reduction algorithms
        - Intelligent edge enhancement
        - Realistic texture synthesis
        - Professional lighting enhancement
        """)
    
    with col_f2:
        st.markdown("""
        *🔧 Processing Methods:*
        - Non-local means denoising
        - Bilateral filtering
        - Adaptive histogram equalization
        - Color space optimization
        - Progressive enhancement iterations
        """)
    
    st.markdown("""
    *Perfect for:* Game asset enhancement, NFT restoration, vintage image restoration, 
    social media content, professional presentations, digital art conversion
    """)
    
    # Example workflow
    st.subheader("📋 How It Works:")
    st.markdown("""
    1. *Upload* your pixel art image
    2. *Choose* enhancement settings and realism mode
    3. *Configure* upscaling factor and quality options
    4. *Generate* the realistic image using AI algorithms
    5. *Download* in multiple quality formats
    """)
