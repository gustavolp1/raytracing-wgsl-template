const THREAD_COUNT = 16;
const RAY_TMIN = 0.0001;
const RAY_TMAX = 100.0;
const PI = 3.1415927f;
const FRAC_1_PI = 0.31830987f;
const FRAC_2_PI = 1.5707964f;

@group(0) @binding(0)  
  var<storage, read_write> fb : array<vec4f>;

@group(0) @binding(1)
  var<storage, read_write> rtfb : array<vec4f>;

@group(1) @binding(0)
  var<storage, read_write> uniforms : array<f32>;

@group(2) @binding(0)
  var<storage, read_write> spheresb : array<sphere>;

@group(2) @binding(1)
  var<storage, read_write> quadsb : array<quad>;

@group(2) @binding(2)
  var<storage, read_write> boxesb : array<box>;

@group(2) @binding(3)
  var<storage, read_write> trianglesb : array<triangle>;

@group(2) @binding(4)
  var<storage, read_write> meshb : array<mesh>;

struct ray {
  origin : vec3f,
  direction : vec3f,
};

struct sphere {
  transform : vec4f,
  color : vec4f,
  material : vec4f,
};

struct quad {
  Q : vec4f,
  u : vec4f,
  v : vec4f,
  color : vec4f,
  material : vec4f,
};

struct box {
  center : vec4f,
  radius : vec4f,
  rotation: vec4f,
  color : vec4f,
  material : vec4f,
};

struct triangle {
  v0 : vec4f,
  v1 : vec4f,
  v2 : vec4f,
};

struct mesh {
  transform : vec4f,
  scale : vec4f,
  rotation : vec4f,
  color : vec4f,
  material : vec4f,
  min : vec4f,
  max : vec4f,
  show_bb : f32,
  start : f32,
  end : f32,
};

struct material_behaviour {
  scatter : bool,
  direction : vec3f,
};

struct camera {
  origin : vec3f,
  lower_left_corner : vec3f,
  horizontal : vec3f,
  vertical : vec3f,
  u : vec3f,
  v : vec3f,
  w : vec3f,
  lens_radius : f32,
};

struct hit_record {
  t : f32,
  p : vec3f,
  normal : vec3f,
  object_color : vec4f,
  object_material : vec4f,
  frontface : bool,
  hit_anything : bool,
};

fn ray_at(r: ray, t: f32) -> vec3f
{
  return r.origin + t * r.direction;
}

fn get_ray(cam: camera, uv: vec2f, rng_state: ptr<function, u32>) -> ray
{
  var rd = cam.lens_radius * rng_next_vec3_in_unit_disk(rng_state);
  var offset = cam.u * rd.x + cam.v * rd.y;
  return ray(cam.origin + offset, normalize(cam.lower_left_corner + uv.x * cam.horizontal + uv.y * cam.vertical - cam.origin - offset));
}

fn get_camera(lookfrom: vec3f, lookat: vec3f, vup: vec3f, vfov: f32, aspect_ratio: f32, aperture: f32, focus_dist: f32) -> camera
{
  var camera = camera();
  camera.lens_radius = aperture / 2.0;

  var theta = degrees_to_radians(vfov);
  var h = tan(theta / 2.0);
  var w = aspect_ratio * h;

  camera.origin = lookfrom;
  camera.w = normalize(lookfrom - lookat);
  camera.u = normalize(cross(vup, camera.w));
  camera.v = cross(camera.u, camera.w);

  camera.lower_left_corner = camera.origin - w * focus_dist * camera.u - h * focus_dist * camera.v - focus_dist * camera.w;
  camera.horizontal = 2.0 * w * focus_dist * camera.u;
  camera.vertical = 2.0 * h * focus_dist * camera.v;

  return camera;
}

fn envoriment_color(direction: vec3f, color1: vec3f, color2: vec3f) -> vec3f
{
  var unit_direction = normalize(direction);
  var t = 0.5 * (unit_direction.y + 1.0);
  var col = (1.0 - t) * color1 + t * color2;

  var sun_direction = normalize(vec3(uniforms[13], uniforms[14], uniforms[15]));
  var sun_color = int_to_rgb(i32(uniforms[17]));
  var sun_intensity = uniforms[16];
  var sun_size = uniforms[18];

  var sun = clamp(dot(sun_direction, unit_direction), 0.0, 1.0);
  col += sun_color * max(0, (pow(sun, sun_size) * sun_intensity));

  return col;
}

fn check_ray_collision(r: ray, max: f32) -> hit_record
{
  var spheresCount = i32(uniforms[19]);
  var quadsCount = i32(uniforms[20]);
  var boxesCount = i32(uniforms[21]);
  var meshCount = i32(uniforms[27]);

  var closest = hit_record(max, vec3f(0.0), vec3f(0.0), vec4f(0.0), vec4f(0.0), false, false);
  var temp_record = hit_record(0.0, vec3f(0.0), vec3f(0.0), vec4f(0.0), vec4f(0.0), false, false);

  // --- 1. Loop through Spheres ---
  for (var i = 0; i < spheresCount; i = i + 1)
  {
    var s = spheresb[i];
    hit_sphere(s.transform.xyz, s.transform.w, r, &temp_record, closest.t);
    
    if (temp_record.hit_anything)
    {
      // hit_sphere is the only function that fully populates
      // t, p, normal, AND frontface.
      closest = temp_record;
      closest.object_color = s.color;
      closest.object_material = s.material;
    }
  }

  // --- 2. Loop through Quads ---
  for (var i = 0; i < quadsCount; i = i + 1)
  {
    var q = quadsb[i];
    hit_quad(r, q.Q, q.u, q.v, &temp_record, closest.t);

    if (temp_record.hit_anything)
    {
      closest.t = temp_record.t;
      closest.p = temp_record.p;
      closest.object_color = q.color;
      closest.object_material = q.material;
      closest.hit_anything = true;

      // FIX: Manually add frontface logic (missing in shapes.wgsl)
      closest.frontface = dot(r.direction, temp_record.normal) < 0.0;
      closest.normal = select(-temp_record.normal, temp_record.normal, closest.frontface);
    }
  }

  // --- 3. Loop through Boxes (A+ Rotated Logic) ---
  for (var i = 0; i < boxesCount; i = i + 1)
  {
    var b = boxesb[i];
    
    // Transform ray into box's local space (handling rotation + translation)
    var inv_rot = q_inverse(b.rotation);
    var local_origin = rotate_vector(r.origin - b.center.xyz, inv_rot);
    var local_dir = rotate_vector(r.direction, inv_rot); // length is 1.0
    var local_ray = ray(local_origin, local_dir);

    // Test the local ray against an AABB centered at (0,0,0)
    hit_box(local_ray, vec3f(0.0), b.radius.xyz, &temp_record, closest.t);

    if (temp_record.hit_anything)
    {
      closest.t = temp_record.t;
      closest.p = ray_at(r, temp_record.t); // Get world-space p
      closest.object_color = b.color;
      closest.object_material = b.material;
      closest.hit_anything = true;

      // Transform normal from local to world space and add frontface logic
      var world_normal = normalize(rotate_vector(temp_record.normal, b.rotation));
      closest.frontface = dot(r.direction, world_normal) < 0.0;
      closest.normal = select(-world_normal, world_normal, closest.frontface);
    }
  }

  // --- 4. Loop through Meshes (B+ and A+ Logic) ---
  for (var i = 0; i < meshCount; i = i + 1)
  {
    var m = meshb[i];
    
    // Transform ray into mesh's local space (T, R, and S)
    var inv_rot = q_inverse(m.rotation);
    var inv_scale_vec = 1.0 / m.scale.xyz;
    var local_origin = rotate_vector(r.origin - m.transform.xyz, inv_rot) * inv_scale_vec;
    
    var local_dir_unnormalized = rotate_vector(r.direction, inv_rot) * inv_scale_vec;
    var local_dir_len = length(local_dir_unnormalized);
    var local_dir = local_dir_unnormalized / local_dir_len;
    var local_ray = ray(local_origin, local_dir);

    // We must scale 't_max' to be in local space
    var local_t_max = closest.t * local_dir_len;

    // Optimization: Check Bounding Box first
    if (AABB_intersect(local_ray, m.min.xyz, m.max.xyz))
    {
      // Loop through all triangles for this mesh
      for (var j = i32(m.start); j < i32(m.end); j = j + 1)
      {
        var tri = trianglesb[j];
        hit_triangle(local_ray, tri.v0.xyz, tri.v1.xyz, tri.v2.xyz, &temp_record, local_t_max);
        
        if (temp_record.hit_anything)
        {
          // Convert hit 't' from local space back to world space
          var t_hit_world = temp_record.t / local_dir_len;
          closest.t = t_hit_world;
          local_t_max = temp_record.t; // Update for next triangle
          
          closest.p = ray_at(r, closest.t);
          closest.object_color = m.color;
          closest.object_material = m.material;
          closest.hit_anything = true;

          // Transform normal from local to world and add frontface logic
          // Normal transform is transpose(inverse(M)), which for T*R*S
          // simplifies to R * S_inv
          var world_normal = normalize(rotate_vector(temp_record.normal * inv_scale_vec, m.rotation));
          closest.frontface = dot(r.direction, world_normal) < 0.0;
          closest.normal = select(-world_normal, world_normal, closest.frontface);
        }
      }
    }
  }

  return closest;
}

fn lambertian(normal : vec3f, absorption: f32, random_sphere: vec3f, rng_state: ptr<function, u32>) -> material_behaviour
{
  // ========== NEW CODE ==========
  // 1. Get the new scatter direction.
  // We add the surface normal to a random vector inside the unit sphere.
  // This creates a cosine-weighted random direction in the normal's hemisphere.
  var scatter_direction = normal + random_sphere;

  // 2. Check for a degenerate (near-zero) ray.
  // This is rare, but if 'random_sphere' was exactly the opposite
  // of 'normal', the result would be a zero vector, which is invalid.
  // If this happens, we just scatter along the normal.
  if (length(scatter_direction) < 0.0001)
  {
    scatter_direction = normal;
  }
  // ========== NEW CODE ==========

  // ========== ALTERED CODE ==========
  // 3. Return the behavior.
  // We're always scattering (true), and the new direction
  // is the normalized scatter_direction.
  return material_behaviour(true, normalize(scatter_direction));
  // ========== ALTERED CODE ==========
}

fn metal(normal : vec3f, direction: vec3f, fuzz: f32, random_sphere: vec3f) -> material_behaviour
{
  // ========== NEW CODE ==========
  // 1. Calculate the perfect reflected direction.
  // 'direction' is the incoming ray's direction.
  // reflect(I, N) = I - 2 * dot(I, N) * N
  var reflected = reflect(direction, normal);

  // 2. Add "fuzz" to the reflection.
  // We add a random vector scaled by the 'fuzz' amount (which is
  // material.specular from the trace function).
  // A fuzz of 0.0 is a perfect mirror.
  // A fuzz of 1.0 is very blurry.
  var scatter_direction = normalize(reflected) + fuzz * random_sphere;

  // 3. Check if the ray scattered "above" the surface.
  // If 'fuzz' is large, the random vector might point the
  // ray "into" the surface. We'll treat this as absorption.
  var scatter = dot(scatter_direction, normal) > 0.0;
  // ========== NEW CODE ==========

  // ========== ALTERED CODE ==========
  if (scatter)
  {
    return material_behaviour(true, normalize(scatter_direction));
  }
  else
  {
    // The ray was absorbed.
    return material_behaviour(false, vec3f(0.0));
  }
  // ========== ALTERED CODE ==========
}

fn dielectric(normal : vec3f, r_direction: vec3f, refraction_index: f32, frontface: bool, random_sphere: vec3f, fuzz: f32, rng_state: ptr<function, u32>) -> material_behaviour
{  
  // ========== NEW CODE ==========
  // 1. Determine the ratio of indices of refraction (n1 / n2).
  // 'refraction_index' (mat_specular) is the IOR of the material. Air is 1.0.
  var refraction_ratio: f32;
  if (frontface) {
    // Ray is in Air (1.0) going into Material (refraction_index)
    refraction_ratio = 1.0 / refraction_index;
  } else {
    // Ray is in Material (refraction_index) going into Air (1.0)
    refraction_ratio = refraction_index;
  }

  var unit_direction = normalize(r_direction);

  // 2. Calculate cosine of the incident angle.
  // We use min() to avoid floating point issues.
  var cos_theta = min(dot(-unit_direction, normal), 1.0);
  
  // 3. Check for Total Internal Reflection (TIR).
  var sin_theta_sq = 1.0 - cos_theta * cos_theta;
  var sin_theta_2_sq = refraction_ratio * refraction_ratio * sin_theta_sq;
  
  var cannot_refract = sin_theta_2_sq > 1.0;
  var scatter_direction: vec3f;

  // 4. Calculate reflectance probability using Schlick's approximation.
  var r0 = (1.0 - refraction_ratio) / (1.0 + refraction_ratio);
  r0 = r0 * r0;
  var reflectance = r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);

  // 5. Decide to REFLECT or REFRACT.
  // We MUST reflect if TIR occurs (cannot_refract).
  // Otherwise, we randomly choose based on the 'reflectance' probability.
  if (cannot_refract || reflectance > rng_next_float(rng_state))
  {
    // Reflect
    scatter_direction = reflect(unit_direction, normal);
  }
  else
  {
    // Refract (calculate the refracted ray direction)
    var perp = refraction_ratio * (unit_direction + cos_theta * normal);
    var parallel = -sqrt(abs(1.0 - sin_theta_2_sq)) * normal;
    scatter_direction = perp + parallel;
  }

  // NOTE: We are ignoring the 'fuzz' parameter (mat_absorption).
  // This implementation is for clear glass. Adding 'fuzz' would
  // create "frosted" glass, which requires more complex logic.
  // ========== NEW CODE ==========

  // ========== ALTERED CODE ==========
  // 6. Return the behavior.
  return material_behaviour(true, normalize(scatter_direction));
  // ========== ALTERED CODE ==========
}

fn emmisive(color: vec3f, light: f32) -> material_behaviour
{
  // ========== ALTERED CODE ==========
  // 1. We don't scatter the ray, so the first parameter is 'false'.
  // 2. We return the object's color multiplied by its light intensity.
  //    The 'trace' function will receive this in the 'behaviour.direction'
  //    field and add it to the final accumulated light.
  return material_behaviour(false, color * light);
  // ========== ALTERED CODE ==========
}

fn trace(r: ray, rng_state: ptr<function, u32>) -> vec3f
{
  // ========== BASE CODE ==========
  var maxbounces = i32(uniforms[2]);
  
  // 'light' is the final color we will return. It accumulates light
  // *emitted* from objects (or the sky).
  var light = vec3f(0.0);

  // 'color' is the attenuation (or throughput). It starts as white (1.0)
  // and gets multiplied by the object's color at each bounce.
  var color = vec3f(1.0);

  // 'r_' is the ray we will bounce around the scene.
  var r_ = r;
  
  var backgroundcolor1 = int_to_rgb(i32(uniforms[11]));
  var backgroundcolor2 = int_to_rgb(i32(uniforms[12]));
  // ========== BASE CODE ==========

  // ========== ALTERED CODE ==========
  // The main bounce loop
  for (var j = 0; j < maxbounces; j = j + 1)
  {
    // 1. See if the ray hits anything in the scene
    var rec = check_ray_collision(r_, RAY_TMAX);

    // 2. If the ray hits *nothing*...
    if (!rec.hit_anything)
    {
      // ...it flies off into the environment (sky).
      // We add the environment's light, modulated by the ray's
      // current attenuation ('color'), to our final 'light'.
      light += color * envoriment_color(r_.direction, backgroundcolor1, backgroundcolor2);
      break; // The ray's life is over.
    }

    // 3. If the ray hits *something*...
    // We get the material properties from the hit record.
    // material = vec4f(smoothness, absorption, specular, light)
    var mat = rec.object_material;
    var mat_smoothness = mat.x; // > 0 metal, < 0 dielectric
    var mat_absorption = mat.y;
    var mat_specular = mat.z;
    var mat_light = mat.w;      // > 0 emissive

    // We get a random vector, which our material functions will use.
    var random_sphere = rng_next_vec3_in_unit_sphere(rng_state);
    var behaviour: material_behaviour; // This will store the material's decision

    // 4. Decide which material function to call.
    
    // --- A. Is it an Emissive (light-emitting) material? ---
    if (mat_light > 0.0)
    {
      behaviour = emmisive(rec.object_color.xyz, mat_light);
      // 'emissive' returns the light it emits in its .direction field.
      // We add this to our final 'light', modulated by attenuation.
      light += color * behaviour.direction;
      break; // Emissive objects don't scatter rays, so we stop here.
    }

    // --- B. Is it a scattering material (Metal, Dielectric, Lambertian)? ---
    if (mat_smoothness > 0.0) // Metal
    {
      behaviour = metal(rec.normal, r_.direction, mat_specular, random_sphere);
    }
    else if (mat_smoothness < 0.0) // Dielectric (glass)
    {
      behaviour = dielectric(rec.normal, r_.direction, mat_specular, rec.frontface, random_sphere, mat_absorption, rng_state);
    }
    else // Lambertian (diffuse)
    {
      behaviour = lambertian(rec.normal, mat_absorption, random_sphere, rng_state);
    }

    // 5. Process the material's decision
    if (!behaviour.scatter)
    {
      // The material decided to absorb the ray (e.g., a metal with
      // 0 reflectivity or a dielectric total internal reflection error).
      break; // The ray's life is over.
    }

    // 6. Prepare for the next bounce
    
    // The ray's attenuation is multiplied by the object's color.
    // (e.g., A white ray hitting a red ball becomes a red ray).
    color *= rec.object_color.xyz;

    // Update the ray for the next loop iteration.
    r_.origin = rec.p;               // The new origin is the hit point.
    r_.direction = behaviour.direction; // The new direction is what the material decided.
  }
  // ========== ALTERED CODE ==========

  // ========== BASE CODE ==========
  return light;
  // ========== BASE CODE ==========
}

@compute @workgroup_size(THREAD_COUNT, THREAD_COUNT, 1)
fn render(@builtin(global_invocation_id) id : vec3u)
{
  // ========== ALTERED CODE ==========
  var rez = uniforms[1];
  var time = u32(uniforms[0]);

  // 1. Initialize RNG
  // We pass the pixel position, resolution, and frame count
  // to get a unique (but repeatable) seed for this pixel.
  var rng_state = init_rng(vec2(id.x, id.y), vec2(u32(rez)), time);

  // 2. Set up Camera
  var lookfrom = vec3(uniforms[7], uniforms[8], uniforms[9]);
  var lookat = vec3(uniforms[23], uniforms[24], uniforms[25]);
  var cam = get_camera(lookfrom, lookat, vec3(0.0, 1.0, 0.0), uniforms[10], 1.0, uniforms[6], uniforms[5]);
  
  // 3. Initialize Color Accumulator
  var color = vec3f(0.0);
  var samples_per_pixel = i32(uniforms[4]);
  var fragCoord = vec2f(f32(id.x), f32(id.y));

  // 4. Main Sample Loop
  // This is for antialiasing. We shoot multiple rays per pixel,
  // each with a tiny random offset, and average the results.
  for (var i = 0; i < samples_per_pixel; i = i + 1)
  {
    // Get a random (u,v) coordinate *within* this pixel
    var uv = (fragCoord + sample_square(&rng_state)) / vec2(rez);

    // Get the ray for this specific (u,v)
    var r = get_ray(cam, uv, &rng_state);

    // Call trace and add the resulting light to our accumulator
    color += trace(r, &rng_state);
  }

  // 5. Average the color
  // If samples_per_pixel is 0, set to 1 to avoid divide-by-zero
  var final_samples = max(1.0, f32(samples_per_pixel));
  color = color / final_samples;

  // 6. Apply Gamma Correction
  // We do this *after* averaging to be physically correct.
  var color_out = vec4(linear_to_gamma(color), 1.0);
  var map_fb = mapfb(id.xy, rez);
  
  // 7. Accumulate Frames
  // This blends the current frame with the previous one
  // to smooth out noise over time.
  var should_accumulate = uniforms[3];
  if (should_accumulate > 0.0)
  {
    // A simple 50/50 blend for accumulation
    color_out = 0.5 * color_out + 0.5 * rtfb[map_fb];
  }

  // 8. Write the final color to the framebuffers
  rtfb[map_fb] = color_out; // This buffer is used for accumulation
  fb[map_fb] = color_out;   // This buffer is rendered to the screen
  // ========== ALTERED CODE ==========
}