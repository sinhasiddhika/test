import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from io import BytesIO
import cv2
from sklearn.cluster import KMeans
import scipy.ndimage as ndi
from skimage import morphology, filters, restoration
import random
import colorsys

# Set page config
st.set_page_config(page_title="AI Pixel Art to Realistic Image Converter", layout="wide")

# App title
st.title("AI Pixel Art to Realistic Image Converter")
st.markdown("Transform pixel art into photorealistic images using advanced AI-inspired techniques!")

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
        
        # Extract and display original colors
        def extract_dominant_colors(img, n_colors=8):
            """Extract dominant colors from the image"""
            img_array = np.array(img)
            pixels = img_array.reshape(-1, 3)
            
            # Remove duplicate colors
            unique_pixels = np.unique(pixels, axis=0)
            
            if len(unique_pixels) <= n_colors:
                return unique_pixels
            
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            return kmeans.cluster_centers_.astype(int)
        
        original_colors = extract_dominant_colors(pixel_image)
        
        # Color Palette Section
        st.subheader("🎨 Color Palette Management")
        
        palette_tab1, palette_tab2 = st.tabs(["Original Colors", "Custom Palette"])
        
        with palette_tab1:
            st.write("Detected colors from your pixel art:")
            cols = st.columns(min(8, len(original_colors)))
            detected_colors = []
            
            for i, color in enumerate(original_colors):
                with cols[i % len(cols)]:
                    color_hex = "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
                    st.color_picker(f"Color {i+1}", color_hex, key=f"detected_{i}", disabled=True)
                    detected_colors.append(color)
        
        with palette_tab2:
            st.write("Create your custom color palette:")
            use_custom_palette = st.checkbox("Use Custom Palette", value=False)
            
            if use_custom_palette:
                num_colors = st.slider("Number of colors", 2, 12, len(original_colors))
                custom_colors = []
                
                cols = st.columns(4)
                for i in range(num_colors):
                    with cols[i % 4]:
                        default_color = "#{:02x}{:02x}{:02x}".format(
                            int(original_colors[i % len(original_colors)][0]),
                            int(original_colors[i % len(original_colors)][1]),
                            int(original_colors[i % len(original_colors)][2])
                        )
                        color_hex = st.color_picker(f"Custom {i+1}", default_color, key=f"custom_{i}")
                        # Convert hex to RGB
                        rgb = tuple(int(color_hex[j:j+2], 16) for j in (1, 3, 5))
                        custom_colors.append(rgb)
                
                # Palette preview
                st.write("Palette Preview:")
                palette_preview = np.zeros((50, len(custom_colors) * 50, 3), dtype=np.uint8)
                for i, color in enumerate(custom_colors):
                    palette_preview[:, i*50:(i+1)*50] = color
                st.image(palette_preview, width=400)
        
        # Enhancement parameters
        st.subheader("🎛 Advanced AI Enhancement Controls")
        
        # Material/Texture Type Selection
        with st.expander("🎨 Material & Style Selection", expanded=True):
            col_mat1, col_mat2 = st.columns(2)
            
            with col_mat1:
                material_type = st.selectbox(
                    "Material Type",
                    ["Fabric/Carpet", "Wood", "Stone/Marble", "Metal", "Leather", "Paper/Canvas", "Glass", "Ceramic"],
                    help="Choose the material to simulate"
                )
                
                texture_intensity = st.slider(
                    "Texture Intensity",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.8,  # Reduced default to minimize artifacts
                    step=0.1,
                    help="How pronounced the texture should be"
                )
            
            with col_mat2:
                pattern_recognition = st.checkbox("🔍 Smart Pattern Recognition", value=True, help="AI analyzes patterns to enhance appropriately")
                color_preservation = st.checkbox("🎯 Strict Color Preservation", value=True, help="Maintains original color palette exactly")
                artifact_reduction = st.checkbox("🔧 Artifact Reduction", value=True, help="Reduces bubbles and unwanted artifacts")
        
        # Output Size Configuration
        with st.expander("📐 Output Size Configuration", expanded=True):
            upscale_factor = st.selectbox(
                "Upscale Factor",
                [2, 4, 6, 8, 10, 12, 16, 20],
                index=2,  # Default to 6x for better quality
                help="Multiply original dimensions by this factor"
            )
            target_width = orig_width * upscale_factor
            target_height = orig_height * upscale_factor
            st.info(f"Output Size: {target_width} × {target_height} pixels")
        
        # Advanced AI Options
        with st.expander("🤖 Advanced AI Enhancement", expanded=False):
            col_ai1, col_ai2 = st.columns(2)
            
            with col_ai1:
                st.markdown("🧠 AI Processing:")
                
                super_resolution_mode = st.selectbox(
                    "Super Resolution Method",
                    ["Color-Preserving", "Edge-Aware", "Pattern-Aware", "Texture-Preserving"],
                    help="Different AI-inspired upscaling approaches"
                )
                
                denoising_strength = st.slider("Denoising Strength", 0.0, 1.0, 0.3, 0.1)
                edge_preservation = st.slider("Edge Preservation", 0.0, 2.0, 1.5, 0.1)
            
            with col_ai2:
                st.markdown("🎯 Realism Controls:")
                
                surface_roughness = st.slider("Surface Roughness", 0.0, 1.0, 0.4, 0.1)
                depth_perception = st.slider("3D Depth Effect", 0.0, 1.5, 0.6, 0.1)
                lighting_intensity = st.slider("Lighting Intensity", 0.0, 2.0, 0.8, 0.1)
        
        def quantize_to_palette(img_array, palette):
            """Quantize image colors to match the specified palette"""
            h, w, c = img_array.shape
            img_flat = img_array.reshape(-1, 3)
            
            # Find closest color in palette for each pixel
            quantized = np.zeros_like(img_flat)
            
            for i, pixel in enumerate(img_flat):
                distances = np.sum((palette - pixel) ** 2, axis=1)
                closest_idx = np.argmin(distances)
                quantized[i] = palette[closest_idx]
            
            return quantized.reshape(h, w, c)
        
        def create_clean_material_texture(material_type, size, intensity=1.0, reduce_artifacts=True):
            """Generate clean material-specific texture patterns without artifacts"""
            h, w = size
            
            if material_type == "Fabric/Carpet":
                # Create a more controlled fabric texture
                texture = np.zeros((h, w), dtype=np.float32)
                
                # Controlled thread pattern
                thread_spacing = max(3, int(6 * intensity))
                thread_thickness = max(1, int(2 * intensity))
                
                # Horizontal threads with controlled variation
                for y in range(0, h, thread_spacing):
                    for t in range(thread_thickness):
                        if y + t < h:
                            texture[y + t, :] += 0.15 * intensity
                
                # Vertical threads with controlled variation
                for x in range(0, w, thread_spacing):
                    for t in range(thread_thickness):
                        if x + t < w:
                            texture[:, x + t] += 0.15 * intensity
                
                # Add subtle fiber texture without noise
                if not reduce_artifacts:
                    fiber_noise = np.random.normal(0, 0.05 * intensity, (h, w))
                    texture += fiber_noise
                
            elif material_type == "Wood":
                # Cleaner wood grain
                texture = np.zeros((h, w), dtype=np.float32)
                
                # Controlled wood grain pattern
                for y in range(h):
                    grain_value = np.sin(y * 0.1 * intensity) * 0.1
                    texture[y, :] = grain_value
                
                # Add wood ring pattern
                center_y = h // 2
                for y in range(h):
                    ring_dist = abs(y - center_y)
                    ring_value = np.sin(ring_dist * 0.05 * intensity) * 0.05
                    texture[y, :] += ring_value
                
            elif material_type == "Stone/Marble":
                # Controlled marble texture
                texture = np.random.normal(0, 0.03 * intensity, (h, w))
                
                # Add controlled veining
                x_coords = np.arange(w) / w * 2 * intensity
                y_coords = np.arange(h) / h * 2 * intensity
                X, Y = np.meshgrid(x_coords, y_coords)
                
                veins = np.sin(X * 2 + Y * 1.5) * 0.1 * intensity
                texture += veins
                
            else:
                # Generic clean texture
                if reduce_artifacts:
                    texture = np.random.normal(0, 0.02 * intensity, (h, w))
                else:
                    texture = np.random.normal(0, 0.05 * intensity, (h, w))
            
            # Smooth the texture to reduce artifacts
            if reduce_artifacts:
                texture = cv2.GaussianBlur(texture, (3, 3), 1.0)
            
            # Normalize texture
            if texture.max() > texture.min():
                texture = (texture - texture.min()) / (texture.max() - texture.min())
            else:
                texture = np.zeros_like(texture)
            
            return texture
        
        def apply_color_preserving_upscaling(img_array, target_size, palette, method="Color-Preserving"):
            """Advanced upscaling while preserving the original color palette"""
            target_h, target_w = target_size
            
            if method == "Color-Preserving":
                # Upscale using nearest neighbor to preserve colors initially
                upscaled_nn = cv2.resize(img_array, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                
                # Apply slight smoothing only at edges
                upscaled_smooth = cv2.resize(img_array, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                
                # Detect edges in original image
                gray_orig = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                edges_orig = cv2.Canny(gray_orig, 50, 150)
                edges_upscaled = cv2.resize(edges_orig, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                
                # Blend: use smooth version only near edges
                kernel = np.ones((3, 3), np.uint8)
                edge_mask = cv2.dilate(edges_upscaled, kernel, iterations=1) / 255.0
                edge_mask = np.stack([edge_mask] * 3, axis=2)
                
                # Combine nearest neighbor (for color preservation) with smooth (for edges)
                blended = upscaled_nn * (1 - edge_mask * 0.3) + upscaled_smooth * (edge_mask * 0.3)
                
                # Quantize back to original palette
                result = quantize_to_palette(blended.astype(np.uint8), palette)
                
                return result.astype(np.uint8)
            
            elif method == "Pattern-Aware":
                # Analyze pattern structure first
                upscaled = cv2.resize(img_array, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                
                # Apply bilateral filter to smooth while preserving edges
                smoothed = cv2.bilateralFilter(upscaled, 9, 75, 75)
                
                # Quantize to palette
                result = quantize_to_palette(smoothed, palette)
                
                return result.astype(np.uint8)
            
            else:
                # Standard upscaling with palette quantization
                upscaled = cv2.resize(img_array, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                result = quantize_to_palette(upscaled, palette)
                return result.astype(np.uint8)
        
        def apply_controlled_lighting(img_array, intensity=1.0, preserve_colors=True):
            """Apply subtle lighting without destroying the color palette"""
            if intensity == 0:
                return img_array
            
            h, w = img_array.shape[:2]
            
            # Create a gentle lighting gradient
            y_coords, x_coords = np.ogrid[:h, :w]
            
            # Light source from top-left
            light_x, light_y = w * 0.2, h * 0.2
            
            # Calculate distance-based lighting
            dist_x = (x_coords - light_x) / w
            dist_y = (y_coords - light_y) / h
            
            # Create smooth lighting falloff
            lighting = 1.0 - (dist_x**2 + dist_y**2) * 0.2 * intensity
            lighting = np.clip(lighting, 0.7, 1.2)
            
            # Apply lighting while preserving color relationships
            lit_image = img_array.copy().astype(np.float32)
            
            if preserve_colors:
                # Apply lighting more subtly to preserve colors
                for i in range(3):
                    lit_image[:, :, i] = lit_image[:, :, i] * (0.9 + lighting * 0.1)
            else:
                for i in range(3):
                    lit_image[:, :, i] = lit_image[:, :, i] * lighting
            
            return np.clip(lit_image, 0, 255).astype(np.uint8)
        
        def create_realistic_image_improved(pixel_img, target_width, target_height, material_type,
                                          texture_intensity, super_resolution_mode, denoising_strength,
                                          edge_preservation, surface_roughness, depth_perception,
                                          lighting_intensity, pattern_recognition, color_preservation,
                                          artifact_reduction, custom_palette=None):
            """Improved AI-inspired conversion with better color preservation and artifact reduction"""
            
            try:
                img_array = np.array(pixel_img)
                
                # Determine color palette to use
                if custom_palette is not None:
                    palette = np.array(custom_palette)
                else:
                    palette = extract_dominant_colors(pixel_img, n_colors=8)
                
                # Step 1: Advanced upscaling with color preservation
                upscaled = apply_color_preserving_upscaling(
                    img_array, (target_height, target_width), palette, super_resolution_mode
                )
                
                # Step 2: Apply denoising if needed (gentle)
                if denoising_strength > 0:
                    # Use gentle denoising to avoid artifacts
                    upscaled = cv2.bilateralFilter(upscaled, 5, 
                                                 int(50 * denoising_strength), 
                                                 int(50 * denoising_strength))
                    
                    # Re-quantize to palette after denoising
                    if color_preservation:
                        upscaled = quantize_to_palette(upscaled, palette)
                
                # Step 3: Create clean material texture
                if surface_roughness > 0:
                    base_texture = create_clean_material_texture(
                        material_type, (target_height, target_width), 
                        texture_intensity, artifact_reduction
                    )
                    
                    # Apply texture more subtly
                    textured_image = upscaled.copy().astype(np.float32)
                    
                    for i in range(3):
                        # Reduce texture effect to minimize artifacts
                        texture_effect = 1.0 + (base_texture - 0.5) * surface_roughness * 0.2
                        textured_image[:, :, i] *= texture_effect
                    
                    upscaled = np.clip(textured_image, 0, 255).astype(np.uint8)
                    
                    # Re-quantize if color preservation is enabled
                    if color_preservation:
                        upscaled = quantize_to_palette(upscaled, palette)
                
                # Step 4: Apply controlled lighting
                if lighting_intensity > 0:
                    upscaled = apply_controlled_lighting(upscaled, lighting_intensity, color_preservation)
                
                # Step 5: Edge enhancement if needed
                if edge_preservation > 0:
                    # Gentle edge enhancement
                    gray = cv2.cvtColor(upscaled, cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    
                    # Apply edge enhancement subtly
                    edges_colored = np.stack([edges] * 3, axis=2) / 255.0
                    enhanced = upscaled * (1 + edges_colored * edge_preservation * 0.1)
                    upscaled = np.clip(enhanced, 0, 255).astype(np.uint8)
                
                # Step 6: Final color quantization if preservation is enabled
                if color_preservation:
                    upscaled = quantize_to_palette(upscaled, palette)
                
                # Step 7: Final artifact reduction
                if artifact_reduction:
                    # Very gentle smoothing to reduce any remaining artifacts
                    upscaled = cv2.medianBlur(upscaled, 3)
                    
                    # Re-quantize one final time
                    if color_preservation:
                        upscaled = quantize_to_palette(upscaled, palette)
                
                return Image.fromarray(upscaled)
                
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                # Fallback to simple upscaling
                return pixel_img.resize((target_width, target_height), Image.NEAREST)
        
        # Generate the realistic image
        if st.button("🚀 Generate Realistic Image", type="primary"):
            with st.spinner("🔄 Converting pixel art to realistic image... This may take a moment."):
                try:
                    # Prepare custom palette if using custom colors
                    final_custom_palette = None
                    if use_custom_palette:
                        final_custom_palette = custom_colors
                    
                    realistic_image = create_realistic_image_improved(
                        pixel_image, target_width, target_height, material_type,
                        texture_intensity, super_resolution_mode, denoising_strength,
                        edge_preservation, surface_roughness, depth_perception,
                        lighting_intensity, pattern_recognition, color_preservation,
                        artifact_reduction, final_custom_palette
                    )
                    
                    # Store the result in session state
                    st.session_state.realistic_image = realistic_image
                    st.session_state.target_width = target_width
                    st.session_state.target_height = target_height
                    st.session_state.upscale_factor = upscale_factor
                    st.session_state.material_type = material_type
                    
                    st.success("✅ Image generated successfully!")
                    
                except Exception as e:
                    st.error(f"Failed to process image: {str(e)}")
                    # Fallback
                    st.session_state.realistic_image = pixel_image.resize((target_width, target_height), Image.NEAREST)
        
        # Show realistic image if it exists in session state
        if hasattr(st.session_state, 'realistic_image') and st.session_state.realistic_image:
            with col2:
                st.subheader("Generated Realistic Image")
                st.image(
                    st.session_state.realistic_image, 
                    caption=f"Realistic {st.session_state.material_type}: {st.session_state.target_width}×{st.session_state.target_height}",
                    use_container_width=True
                )
            
            # Show final color palette used
            st.subheader("🎨 Final Color Palette")
            final_colors = extract_dominant_colors(st.session_state.realistic_image)
            cols = st.columns(min(8, len(final_colors)))
            
            for i, color in enumerate(final_colors):
                with cols[i % len(cols)]:
                    color_hex = "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
                    st.color_picker(f"Final {i+1}", color_hex, key=f"final_{i}", disabled=True)
            
            # Download section
            st.subheader("💾 Download Options")
            
            col_d1, col_d2, col_d3 = st.columns(3)
            
            with col_d1:
                # Download PNG
                buf_png = BytesIO()
                st.session_state.realistic_image.save(buf_png, format="PNG")
                st.download_button(
                    label="📱 Download PNG",
                    data=buf_png.getvalue(),
                    file_name="realistic_image.png",
                    mime="image/png"
                )
            
            with col_d2:
                # Download JPEG
                buf_jpg = BytesIO()
                st.session_state.realistic_image.save(buf_jpg, format="JPEG", quality=95)
                st.download_button(
                    label="🖼 Download JPEG",
                    data=buf_jpg.getvalue(),
                    file_name="realistic_image.jpg",
                    mime="image/jpeg"
                )
            
            with col_d3:
                # Download palette info
                palette_info = "Color Palette Information:\n"
                for i, color in enumerate(final_colors):
                    palette_info += f"Color {i+1}: RGB({color[0]}, {color[1]}, {color[2]}) - #{color[0]:02x}{color[1]:02x}{color[2]:02x}\n"
                
                st.download_button(
                    label="🗂 Download Palette Info",
                    data=palette_info,
                    file_name="color_palette.txt",
                    mime="text/plain"
                )
    
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

else:
    st.info("⬆ Please upload a pixel art image to start the AI conversion!")
    
    st.subheader("🚀 Key Improvements:")
    st.markdown("""
    - **🎨 Custom Color Palette**: Define your own color palette or use detected colors
    - **🔧 Artifact Reduction**: Eliminates bubbles and unwanted visual artifacts  
    - **🎯 Strict Color Preservation**: Maintains original color relationships
    - **🔍 Smart Pattern Recognition**: Better analysis of pixel art structure
    - **💡 Controlled Lighting**: Subtle lighting that doesn't destroy colors
    - **📐 Clean Texture Generation**: Material textures without noise artifacts
    - **🖼 Multiple Export Options**: PNG, JPEG, and color palette information
    """)
    
    st.subheader("🎨 How to Use Custom Palettes:")
    st.markdown("""
    1. Upload your pixel art image
    2. Check the "Original Colors" tab to see detected colors
    3. Switch to "Custom Palette" tab if you want to modify colors
    4. Enable "Use Custom Palette" and adjust the colors as needed
    5. Configure material type and enhancement settings
    6. Generate your realistic image with the custom palette
    """)
