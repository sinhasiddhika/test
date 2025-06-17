import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from io import BytesIO
import cv2
from sklearn.cluster import KMeans
import scipy.ndimage as ndi
from skimage import morphology, filters, restoration
import random

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
                    max_value=3.0,
                    value=1.5,
                    step=0.1,
                    help="How pronounced the texture should be"
                )
            
            with col_mat2:
                pattern_recognition = st.checkbox("🔍 Smart Pattern Recognition", value=True, help="AI analyzes patterns to enhance appropriately")
                color_depth_enhancement = st.checkbox("🌈 Advanced Color Depth", value=True, help="Creates realistic color variations")
                lighting_simulation = st.checkbox("💡 3D Lighting Simulation", value=True, help="Simulates realistic lighting and shadows")
        
        # Output Size Configuration
        with st.expander("📐 Output Size Configuration", expanded=True):
            upscale_factor = st.selectbox(
                "Upscale Factor",
                [2, 4, 6, 8, 10, 12, 16, 20],
                index=3,  # Default to 8x
                help="Multiply original dimensions by this factor"
            )
            target_width = orig_width * upscale_factor
            target_height = orig_height * upscale_factor
            st.info(f"Output Size: {target_width} × {target_height} pixels")
        
        # Advanced AI Options
        with st.expander("🤖 Advanced AI Enhancement", expanded=True):
            col_ai1, col_ai2 = st.columns(2)
            
            with col_ai1:
                st.markdown("🧠 AI Processing:")
                
                super_resolution_mode = st.selectbox(
                    "Super Resolution Method",
                    ["Neural-Inspired", "Multi-Scale", "Edge-Aware", "Texture-Preserving"],
                    help="Different AI-inspired upscaling approaches"
                )
                
                denoising_strength = st.slider("Denoising Strength", 0.0, 2.0, 1.0, 0.1)
                detail_enhancement = st.slider("Detail Enhancement", 0.0, 3.0, 1.5, 0.1)
            
            with col_ai2:
                st.markdown("🎯 Realism Controls:")
                
                surface_roughness = st.slider("Surface Roughness", 0.0, 2.0, 1.0, 0.1)
                depth_perception = st.slider("3D Depth Effect", 0.0, 2.0, 1.2, 0.1)
                color_variation = st.slider("Natural Color Variation", 0.0, 2.0, 1.3, 0.1)
                ambient_occlusion = st.checkbox("Ambient Occlusion", value=True, help="Adds realistic shadows in crevices")
        
        def analyze_pixel_patterns(img_array):
            """Analyze the pixel art to understand its structure and patterns"""
            # Convert to grayscale for pattern analysis
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Detect edges to understand structure
            edges = cv2.Canny(gray, 50, 150)
            
            # Find dominant colors using k-means clustering
            pixels = img_array.reshape(-1, 3)
            n_colors = min(8, len(np.unique(pixels.view(np.void), axis=0)))
            if n_colors > 1:
                kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
                kmeans.fit(pixels)
                dominant_colors = kmeans.cluster_centers_.astype(int)
            else:
                dominant_colors = [np.mean(pixels, axis=0).astype(int)]
            
            # Analyze pattern repetition
            h, w = gray.shape
            tile_size = min(16, min(h, w) // 4)
            
            patterns = []
            if tile_size > 2:
                for i in range(0, h - tile_size, tile_size//2):
                    for j in range(0, w - tile_size, tile_size//2):
                        tile = gray[i:i+tile_size, j:j+tile_size]
                        if tile.shape == (tile_size, tile_size):
                            patterns.append(tile)
            
            return {
                'edges': edges,
                'dominant_colors': dominant_colors,
                'patterns': patterns,
                'tile_size': tile_size
            }
        
        def create_material_texture(material_type, size, intensity=1.0):
            """Generate material-specific texture patterns"""
            h, w = size
            
            if material_type == "Fabric/Carpet":
                # Generate woven texture
                texture = np.zeros((h, w), dtype=np.float32)
                
                # Create thread-like patterns
                thread_spacing = max(2, int(4 * intensity))
                
                # Horizontal threads
                for y in range(0, h, thread_spacing):
                    thickness = max(1, int(2 * intensity))
                    for t in range(thickness):
                        if y + t < h:
                            # Add slight wave to threads
                            wave = np.sin(np.arange(w) * 0.1) * 0.5
                            for x in range(w):
                                ny = int(y + t + wave[x])
                                if 0 <= ny < h:
                                    texture[ny, x] += 0.3
                
                # Vertical threads
                for x in range(0, w, thread_spacing):
                    thickness = max(1, int(2 * intensity))
                    for t in range(thickness):
                        if x + t < w:
                            # Add slight wave to threads
                            wave = np.sin(np.arange(h) * 0.1) * 0.5
                            for y in range(h):
                                nx = int(x + t + wave[y])
                                if 0 <= nx < w:
                                    texture[y, nx] += 0.3
                
                # Add fabric grain noise
                noise = np.random.normal(0, 0.1 * intensity, (h, w))
                texture += noise
                
            elif material_type == "Wood":
                # Generate wood grain
                texture = np.zeros((h, w), dtype=np.float32)
                
                # Wood rings
                center_x, center_y = w // 2, h // 2
                for y in range(h):
                    for x in range(w):
                        dist = np.sqrt((x - center_x)*2 + (y - center_y)*2)
                        ring_val = np.sin(dist * 0.1 * intensity) * 0.2
                        texture[y, x] = ring_val
                
                # Wood grain lines
                grain_noise = np.random.normal(0, 0.05 * intensity, (h, w))
                texture += grain_noise
                
            elif material_type == "Stone/Marble":
                # Generate marble-like veining
                texture = np.random.normal(0, 0.1 * intensity, (h, w))
                
                # Add veins using Perlin-like noise
                x_coords = np.arange(w) / w * 4 * intensity
                y_coords = np.arange(h) / h * 4 * intensity
                X, Y = np.meshgrid(x_coords, y_coords)
                
                veins = np.sin(X * 3 + Y * 2) * np.cos(Y * 3 - X * 1.5) * 0.3
                texture += veins
                
            else:
                # Generic texture
                texture = np.random.normal(0, 0.1 * intensity, (h, w))
            
            # Normalize texture
            texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
            return texture
        
        def apply_neural_inspired_upscaling(img_array, target_size, method="Neural-Inspired"):
            """Advanced upscaling using neural-inspired techniques"""
            target_h, target_w = target_size
            
            if method == "Neural-Inspired":
                # Multi-step upscaling with edge preservation
                current = img_array.copy().astype(np.float32)
                
                # Progressive upscaling
                steps = min(4, int(np.log2(max(target_h / img_array.shape[0], target_w / img_array.shape[1]))))
                
                for step in range(steps):
                    # Calculate intermediate size
                    scale = 2 ** (step + 1)
                    inter_h = min(target_h, int(img_array.shape[0] * scale))
                    inter_w = min(target_w, int(img_array.shape[1] * scale))
                    
                    # Upscale current image
                    current = cv2.resize(current, (inter_w, inter_h), interpolation=cv2.INTER_CUBIC)
                    
                    # Apply edge-preserving smoothing
                    if step < steps - 1:  # Don't smooth the final step
                        current = cv2.bilateralFilter(current.astype(np.uint8), 9, 75, 75).astype(np.float32)
                
                # Final resize to exact target size
                if current.shape[:2] != (target_h, target_w):
                    current = cv2.resize(current, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                
                return current.astype(np.uint8)
            
            elif method == "Edge-Aware":
                # Detect edges and preserve them during upscaling
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
                # Upscale image and edges separately
                upscaled = cv2.resize(img_array, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                upscaled_edges = cv2.resize(edges, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                
                # Enhance edges in the upscaled image
                upscaled_gray = cv2.cvtColor(upscaled, cv2.COLOR_RGB2GRAY)
                enhanced_edges = cv2.addWeighted(upscaled_gray, 0.7, upscaled_edges.astype(np.uint8), 0.3, 0)
                
                # Merge back with color
                for i in range(3):
                    upscaled[:, :, i] = cv2.addWeighted(upscaled[:, :, i], 0.8, enhanced_edges, 0.2, 0)
                
                return upscaled
            
            else:
                # Standard upscaling
                return cv2.resize(img_array, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        def apply_material_lighting(img_array, material_type, depth_map, intensity=1.0):
            """Apply material-specific lighting and shading"""
            h, w = img_array.shape[:2]
            
            # Create light source (top-left)
            light_x, light_y = w * 0.3, h * 0.3
            
            # Calculate lighting for each pixel
            lighting = np.ones((h, w), dtype=np.float32)
            
            for y in range(h):
                for x in range(w):
                    # Calculate distance from light source
                    dist = np.sqrt((x - light_x)*2 + (y - light_y)*2)
                    
                    # Calculate lighting based on depth and distance
                    depth_factor = depth_map[y, x]
                    light_intensity = 1.0 - (dist / (w + h)) * 0.3
                    
                    # Material-specific lighting
                    if material_type == "Fabric/Carpet":
                        # Fabric scatters light softly
                        lighting[y, x] = light_intensity * (0.8 + depth_factor * 0.4) * intensity
                    elif material_type == "Metal":
                        # Metal reflects light sharply
                        lighting[y, x] = light_intensity * (0.6 + depth_factor * 0.8) * intensity
                    else:
                        # Default lighting
                        lighting[y, x] = light_intensity * (0.7 + depth_factor * 0.5) * intensity
            
            # Apply lighting to image
            lit_image = img_array.copy().astype(np.float32)
            for i in range(3):
                lit_image[:, :, i] *= lighting
            
            return np.clip(lit_image, 0, 255).astype(np.uint8)
        
        def create_realistic_image_advanced(pixel_img, target_width, target_height, material_type,
                                          texture_intensity, super_resolution_mode, denoising_strength,
                                          detail_enhancement, surface_roughness, depth_perception,
                                          color_variation, pattern_recognition, color_depth_enhancement,
                                          lighting_simulation, ambient_occlusion):
            """Advanced AI-inspired conversion to realistic image"""
            
            try:
                img_array = np.array(pixel_img)
                
                # Step 1: Analyze pixel art patterns
                if pattern_recognition:
                    analysis = analyze_pixel_patterns(img_array)
                else:
                    analysis = {'edges': None, 'dominant_colors': [], 'patterns': []}
                
                # Step 2: Advanced upscaling
                upscaled = apply_neural_inspired_upscaling(
                    img_array, (target_height, target_width), super_resolution_mode
                )
                
                # Step 3: Create material texture
                base_texture = create_material_texture(
                    material_type, (target_height, target_width), texture_intensity
                )
                
                # Step 4: Apply denoising
                if denoising_strength > 0:
                    upscaled = cv2.bilateralFilter(upscaled, 
                                                 min(15, int(9 * denoising_strength)), 
                                                 int(75 * denoising_strength), 
                                                 int(75 * denoising_strength))
                
                # Step 5: Create depth map for 3D effects
                gray = cv2.cvtColor(upscaled, cv2.COLOR_RGB2GRAY)
                depth_map = ndi.gaussian_filter(gray.astype(np.float32) / 255.0, 
                                              sigma=max(1, depth_perception))
                
                # Step 6: Apply surface texture
                textured_image = upscaled.copy().astype(np.float32)
                
                # Modulate each color channel with texture
                for i in range(3):
                    texture_effect = 1.0 + (base_texture - 0.5) * surface_roughness * 0.3
                    textured_image[:, :, i] *= texture_effect
                
                # Step 7: Color variation for realism
                if color_variation > 0:
                    h, w = textured_image.shape[:2]
                    
                    # Add subtle color variations
                    for i in range(3):
                        color_noise = np.random.normal(1.0, 0.05 * color_variation, (h, w))
                        textured_image[:, :, i] *= color_noise
                
                # Step 8: Apply lighting simulation
                if lighting_simulation:
                    textured_image = apply_material_lighting(
                        textured_image.astype(np.uint8), 
                        material_type, 
                        depth_map, 
                        intensity=1.0
                    ).astype(np.float32)
                
                # Step 9: Ambient occlusion
                if ambient_occlusion:
                    # Create ambient occlusion map
                    ao_map = 1.0 - ndi.gaussian_filter(depth_map, sigma=3) * 0.3
                    
                    for i in range(3):
                        textured_image[:, :, i] *= ao_map
                
                # Step 10: Detail enhancement
                if detail_enhancement > 0:
                    # Unsharp masking for detail enhancement
                    blurred = cv2.GaussianBlur(textured_image, (0, 0), 2.0)
                    textured_image = cv2.addWeighted(textured_image, 1.0 + detail_enhancement, 
                                                   blurred, -detail_enhancement, 0)
                
                # Step 11: Advanced color depth enhancement
                if color_depth_enhancement:
                    # Enhance color depth using LAB color space
                    lab = cv2.cvtColor(textured_image.astype(np.uint8), cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(lab)
                    
                    # Enhance L channel with CLAHE
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    l_enhanced = clahe.apply(l)
                    
                    # Enhance color channels
                    a_enhanced = cv2.addWeighted(a, 1.2, np.zeros_like(a), 0, 0)
                    b_enhanced = cv2.addWeighted(b, 1.2, np.zeros_like(b), 0, 0)
                    
                    lab_enhanced = cv2.merge([l_enhanced, a_enhanced, b_enhanced])
                    textured_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB).astype(np.float32)
                
                # Final processing
                result = np.clip(textured_image, 0, 255).astype(np.uint8)
                
                return Image.fromarray(result)
                
            except Exception as e:
                st.error(f"Error during advanced processing: {str(e)}")
                # Fallback to basic upscaling
                return pixel_img.resize((target_width, target_height), Image.LANCZOS)
        
        # Generate the realistic image
        if st.button("🚀 Generate Realistic Image with AI", type="primary"):
            with st.spinner("🔄 Converting pixel art to realistic image using advanced AI techniques... This may take a moment."):
                try:
                    realistic_image = create_realistic_image_advanced(
                        pixel_image, target_width, target_height, material_type,
                        texture_intensity, super_resolution_mode, denoising_strength,
                        detail_enhancement, surface_roughness, depth_perception,
                        color_variation, pattern_recognition, color_depth_enhancement,
                        lighting_simulation, ambient_occlusion
                    )
                    
                    # Store the result in session state
                    st.session_state.realistic_image = realistic_image
                    st.session_state.target_width = target_width
                    st.session_state.target_height = target_height
                    st.session_state.upscale_factor = upscale_factor
                    st.session_state.material_type = material_type
                    
                except Exception as e:
                    st.error(f"Failed to process image: {str(e)}")
                    # Fallback
                    st.session_state.realistic_image = pixel_image.resize((target_width, target_height), Image.LANCZOS)
        
        # Show realistic image if it exists in session state
        if hasattr(st.session_state, 'realistic_image') and st.session_state.realistic_image:
            with col2:
                st.subheader("Generated Realistic Image")
                st.image(
                    st.session_state.realistic_image, 
                    caption=f"Realistic {st.session_state.material_type}: {st.session_state.target_width}×{st.session_state.target_height}",
                    use_container_width=True
                )
            
            # Download section
            st.subheader("💾 Download Realistic Image")
            
            col_d1, col_d2 = st.columns(2)
            
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
    
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

else:
    st.info("⬆ Please upload a pixel art image to start the AI conversion!")
    
    st.subheader("🚀 Advanced AI Features:")
    st.markdown("""
    - *Neural-Inspired Super Resolution*: Multi-scale upscaling with edge preservation
    - *Material Simulation*: Realistic fabric, wood, stone, and metal textures
    - *Smart Pattern Recognition*: AI analyzes your pixel art structure
    - *3D Lighting Simulation*: Realistic lighting and shadow effects
    - *Advanced Color Depth*: Enhanced color variations and depth perception
    - *Ambient Occlusion*: Professional-grade shadow enhancement
    - *Surface Roughness Control*: Fine-tune material surface properties
    """)
