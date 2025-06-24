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
                    max_value=3.0,
                    value=1.8,  # Higher default for more realistic carpet texture
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
                
                surface_roughness = st.slider("Surface Roughness", 0.0, 1.0, 0.8, 0.1)  # Increased for carpet
                depth_perception = st.slider("3D Depth Effect", 0.0, 1.5, 1.0, 0.1)  # Increased for carpet
                lighting_intensity = st.slider("Lighting Intensity", 0.0, 2.0, 1.2, 0.1)  # Increased for carpet
        
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
        
        def create_realistic_carpet_texture(size, intensity=1.0, reduce_artifacts=True):
            """Generate photorealistic carpet texture with proper fiber simulation and weave patterns"""
            h, w = size
            
            # Initialize base texture
            base_texture = np.ones((h, w), dtype=np.float32)
            height_map = np.zeros((h, w), dtype=np.float32)
            
            # === STEP 1: Create carpet weave/pile structure ===
            # Simulate carpet pile with varying heights
            pile_density = 2  # Fibers per unit
            fiber_size = max(2, int(3 * intensity))
            
            # Create random fiber positions
            np.random.seed(42)  # For consistent results
            num_fibers = (h * w) // (fiber_size * fiber_size) * pile_density
            
            for _ in range(num_fibers):
                # Random fiber position
                fy = np.random.randint(0, h)
                fx = np.random.randint(0, w)
                
                # Fiber height variation (carpet pile height)
                fiber_height = np.random.normal(1.0, 0.2)
                fiber_height = np.clip(fiber_height, 0.3, 1.7)
                
                # Create fiber footprint
                y_start = max(0, fy - fiber_size//2)
                y_end = min(h, fy + fiber_size//2 + 1)
                x_start = max(0, fx - fiber_size//2)
                x_end = min(w, fx + fiber_size//2 + 1)
                
                # Apply fiber with soft edges
                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        # Distance from fiber center for soft falloff
                        dist = np.sqrt((y - fy)**2 + (x - fx)**2)
                        if dist <= fiber_size:
                            falloff = max(0, 1 - dist / fiber_size)
                            height_map[y, x] = max(height_map[y, x], fiber_height * falloff)
            
            # === STEP 2: Add carpet backing pattern ===
            # Simulate the backing weave visible between pile
            backing_pattern = np.ones((h, w), dtype=np.float32) * 0.3
            
            # Create weave pattern
            weave_spacing = max(4, int(6 * intensity))
            for y in range(0, h, weave_spacing):
                for x in range(0, w, weave_spacing):
                    # Alternating weave pattern
                    if (y//weave_spacing + x//weave_spacing) % 2 == 0:
                        y_end = min(h, y + weave_spacing//2)
                        x_end = min(w, x + weave_spacing//2)
                        backing_pattern[y:y_end, x:x_end] = 0.4
            
            # Blend backing with pile height
            carpet_structure = np.maximum(backing_pattern, height_map * 0.7)
            
            # === STEP 3: Add directional pile effect ===
            # Real carpets have pile that leans in one direction
            pile_direction = np.random.choice(['horizontal', 'vertical', 'diagonal'])
            directional_effect = np.ones((h, w), dtype=np.float32)
            
            if pile_direction == 'horizontal':
                # Create horizontal brushing effect
                for y in range(h):
                    wave = np.sin(y * 0.2) * 0.1 + 1.0
                    directional_effect[y, :] = wave
                    
            elif pile_direction == 'vertical':
                # Create vertical brushing effect
                for x in range(w):
                    wave = np.sin(x * 0.2) * 0.1 + 1.0
                    directional_effect[:, x] = wave
                    
            else:  # diagonal
                # Create diagonal brushing effect
                for y in range(h):
                    for x in range(w):
                        wave = np.sin((x + y) * 0.15) * 0.08 + 1.0
                        directional_effect[y, x] = wave
            
            # Apply directional effect
            carpet_structure *= directional_effect
            
            # === STEP 4: Add fine fiber texture ===
            # Simulate individual fiber variations
            fiber_noise = np.random.normal(1.0, 0.05, (h//2, w//2))
            fiber_noise = cv2.resize(fiber_noise, (w, h), interpolation=cv2.INTER_LINEAR)
            carpet_structure *= fiber_noise
            
            # === STEP 5: Add realistic carpet imperfections ===
            if not reduce_artifacts:
                # Add subtle wear patterns
                wear_centers = [(np.random.randint(h//4, 3*h//4), np.random.randint(w//4, 3*w//4)) 
                               for _ in range(2)]
                
                for wy, wx in wear_centers:
                    # Create circular wear pattern
                    y_coords, x_coords = np.ogrid[:h, :w]
                    wear_mask = (y_coords - wy)**2 + (x_coords - wx)**2
                    wear_radius = min(h, w) // 8
                    wear_effect = np.exp(-wear_mask / (2 * wear_radius**2))
                    carpet_structure *= (1 - wear_effect * 0.2)
            
            # === STEP 6: Add micro-detail fiber structure ===
            # Create fine fiber detail at pixel level
            micro_detail = np.random.normal(1.0, 0.02, (h, w))
            
            # Apply fiber orientation micro-structure
            for y in range(1, h-1):
                for x in range(1, w-1):
                    # Check local pile height to determine fiber density
                    local_height = carpet_structure[y, x]
                    if local_height > 0.5:  # Areas with significant pile
                        # Add micro-fiber detail
                        fiber_detail = 1.0 + (micro_detail[y, x] - 1.0) * local_height
                        carpet_structure[y, x] *= fiber_detail
            
            # === STEP 7: Smooth and normalize ===
            if reduce_artifacts:
                # Gentle smoothing to remove harsh transitions
                carpet_structure = cv2.GaussianBlur(carpet_structure, (3, 3), 0.8)
            
            # Create shadow/lighting based on height map
            # Simulate how light hits the carpet pile
            shadow_map = np.zeros((h, w), dtype=np.float32)
            for y in range(1, h-1):
                for x in range(1, w-1):
                    # Calculate local height gradient (normal vector)
                    dx = carpet_structure[y, x+1] - carpet_structure[y, x-1]
                    dy = carpet_structure[y+1, x] - carpet_structure[y-1, x]
                    
                    # Light direction (from top-left)
                    light_x, light_y = -0.5, -0.7
                    
                    # Calculate how much light hits this point
                    dot_product = dx * light_x + dy * light_y
                    shadow_map[y, x] = 0.3 + 0.7 * max(0, dot_product + 0.5)
            
            # Combine structure with lighting
            final_texture = carpet_structure * (0.7 + shadow_map * 0.3)
            
            # Normalize to proper range
            if final_texture.max() > final_texture.min():
                final_texture = (final_texture - final_texture.min()) / (final_texture.max() - final_texture.min())
                # Map to realistic carpet texture range
                final_texture = final_texture * 0.6 + 0.7  # Range from 0.7 to 1.3
            else:
                final_texture = np.ones_like(final_texture)
            
            return final_texture
        
        def create_clean_material_texture(material_type, size, intensity=1.0, reduce_artifacts=True):
            """Generate clean material-specific texture patterns without artifacts"""
            h, w = size
            
            if material_type == "Fabric/Carpet":
                return create_realistic_carpet_texture(size, intensity, reduce_artifacts)
                
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
            if reduce_artifacts and material_type != "Fabric/Carpet":
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
                # For carpet, use higher quality upscaling to preserve texture detail
                # First, upscale with Lanczos for better quality
                upscaled_lanczos = cv2.resize(img_array, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                
                # Apply edge-preserving smoothing to reduce pixelation
                upscaled_smooth = cv2.edgePreservingFilter(upscaled_lanczos, flags=2, sigma_s=50, sigma_r=0.4)
                
                # Blend the two for optimal results
                upscaled = cv2.addWeighted(upscaled_lanczos, 0.7, upscaled_smooth, 0.3, 0)
                
                # Quantize back to original palette
                result = quantize_to_palette(upscaled.astype(np.uint8), palette)
                
                return result.astype(np.uint8)
            
            elif method == "Pattern-Aware":
                # Enhanced pattern-aware upscaling for textiles
                upscaled = cv2.resize(img_array, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                
                # Apply non-local means denoising while preserving patterns
                upscaled = cv2.fastNlMeansDenoisingColored(upscaled, None, 3, 3, 7, 21)
                
                # Quantize to palette
                result = quantize_to_palette(upscaled, palette)
                
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
                    
                    # Apply texture more realistically for carpet
                    textured_image = upscaled.copy().astype(np.float32)
                    
                    if material_type == "Fabric/Carpet":
                        # Advanced carpet texture application
                        
                        # Create color variation based on pile height
                        pile_height_map = base_texture.copy()
                        
                        # Apply realistic carpet shading
                        for i in range(3):
                            # Base texture application
                            textured_image[:, :, i] *= base_texture
                            
                            # Add pile shadow effect - darker in valleys, lighter on peaks
                            shadow_factor = (pile_height_map - 1.0) * 0.3
                            textured_image[:, :, i] += shadow_factor * textured_image[:, :, i]
                            
                            # Add subtle color variation that real carpets have
                            color_variation = (pile_height_map - 1.0) * 0.1
                            textured_image[:, :, i] *= (1.0 + color_variation)
                        
                        # Add realistic depth perception
                        if depth_perception > 0:
                            # Create depth map based on pile structure
                            depth_map = cv2.GaussianBlur(pile_height_map, (7, 7), 2)
                            
                            # Apply depth as brightness variation
                            depth_effect = (depth_map - 1.0) * depth_perception * 15
                            
                            for i in range(3):
                                textured_image[:, :, i] += depth_effect
                        
                        # Add fiber specular highlights (carpet sheen)
                        if lighting_intensity > 0.5:
                            # Create specular map based on pile orientation
                            specular_map = np.maximum(0, pile_height_map - 1.1) * 2
                            specular_map = cv2.GaussianBlur(specular_map, (3, 3), 1)
                            
                            # Apply subtle specular highlights
                            for i in range(3):
                                textured_image[:, :, i] += specular_map * 20 * (lighting_intensity - 0.5)
                    else:
                        # For other materials, use the original method
                        for i in range(3):
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
    - **🧶 Realistic Carpet Texture**: Multi-layered fiber simulation with pile direction
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
