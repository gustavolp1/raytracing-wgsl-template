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

fn quaternion_rotation(v: vec3<f32>, q_in: vec4<f32>) -> vec3<f32> {
    let q = q_in / max(length(q_in), 1e-8);
    let u = q.xyz;
    let s = q.w;

    return 2.0 * dot(u, v) * u +
           (s * s - dot(u, u)) * v +
           2.0 * s * cross(u, v);
}

fn check_ray_collision(r: ray, max: f32) -> hit_record
{
  var spheresCount = i32(uniforms[19]);
  var quadsCount = i32(uniforms[20]);
  var boxesCount = i32(uniforms[21]);
  var meshCount = i32(uniforms[27]); // This is the count we need
  var trianglesCount = i32(uniforms[22]); // This is NOT for the mesh loop

  var closest = hit_record(max, vec3f(0.0), vec3f(0.0), vec4f(0.0), vec4f(0.0), false, false);
  var temp_record = hit_record(0.0, vec3f(0.0), vec3f(0.0), vec4f(0.0), vec4f(0.0), false, false);

  // --- 1. Loop through Spheres ---
  for (var i = 0; i < spheresCount; i = i + 1)
  {
    var s = spheresb[i];
    hit_sphere(s.transform.xyz, s.transform.w, r, &temp_record, closest.t);
    
    if (temp_record.hit_anything)
    {
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

      closest.frontface = dot(r.direction, temp_record.normal) < 0.0;
      closest.normal = select(-temp_record.normal, temp_record.normal, closest.frontface);
    }
  }

  // --- 3. Loop through Boxes ---
  for (var i = 0; i < boxesCount; i = i + 1)
  {
    var b = boxesb[i];
    var rot_q = quaternion_from_euler(b.rotation.xyz);
    var inv_rot = q_inverse(rot_q);
    var local_origin = rotate_vector(r.origin - b.center.xyz, inv_rot);
    var local_dir = rotate_vector(r.direction, inv_rot);
    var local_ray = ray(local_origin, local_dir);

    hit_box(local_ray, vec3f(0.0), b.radius.xyz, &temp_record, closest.t);

    if (temp_record.hit_anything)
    {
      closest.t = temp_record.t;
      closest.p = ray_at(r, temp_record.t); 
      closest.object_color = b.color;
      closest.object_material = b.material;
      closest.hit_anything = true;

      var world_normal = normalize(rotate_vector(temp_record.normal, rot_q));
      closest.frontface = dot(r.direction, world_normal) < 0.0;
      closest.normal = select(-world_normal, world_normal, closest.frontface);
    }
  }

  // --- 4. Loop through Meshes ---
  
  // BUGFIX 1: Loop over 'meshCount', not 'trianglesCount'
  for (var i = 0; i < meshCount; i = i + 1) {
    var curr_mesh = meshb[i];

    var rot = curr_mesh.rotation.xyz;
    let q = quaternion_from_euler(rot.xyz);
    var transform = curr_mesh.transform.xyz;
    var scale = curr_mesh.scale.xyz;

    var bound_min = quaternion_rotation(curr_mesh.min.xyz * scale, q) + transform;
    var bound_max = quaternion_rotation(curr_mesh.max.xyz * scale, q) + transform;

    var inside = AABB_intersect(r, bound_min, bound_max);
    
    if (inside){
      
      var triangle_start = curr_mesh.start;
      var triangle_end = curr_mesh.end;

      var color = curr_mesh.color;
      var material = curr_mesh.material;

      for (var j = i32(triangle_start); j < i32(triangle_end); j = j + 1){

        var curr_tri = trianglesb[j];

        let v0_new = quaternion_rotation(curr_tri.v0.xyz * scale, q) + transform;
        let v1_new = quaternion_rotation(curr_tri.v1.xyz * scale, q) + transform;
        let v2_new = quaternion_rotation(curr_tri.v2.xyz * scale, q) + transform;

        // BUGFIX 2: DELETED the 3 lines that wrote back to 'curr_tri'
        
        // Use the transformed vertices (v0_new, v1_new, v2_new) directly
        hit_triangle(r, v0_new, v1_new, v2_new, &temp_record, closest.t);

        if (!temp_record.hit_anything) {
          continue;
        }

        temp_record.object_material = material;
        temp_record.object_color = color;

        closest = temp_record;
      } 
    }
  }

  return closest;
}

fn lambertian(normal : vec3f, absorption: f32, random_sphere: vec3f, rng_state: ptr<function, u32>) -> material_behaviour
{
  var scatter_direction = normal + random_sphere;

  if (length(scatter_direction) < 0.0001)
  {
    scatter_direction = normal;
  }
  return material_behaviour(true, normalize(scatter_direction));
}

fn metal(normal : vec3f, direction: vec3f, fuzz: f32, random_sphere: vec3f) -> material_behaviour
{
  var reflected = reflect(direction, normal);

  var scatter_direction = normalize(reflected) + fuzz * random_sphere;

  var scatter = dot(scatter_direction, normal) > 0.0;

  if (scatter)
  {
    return material_behaviour(true, normalize(scatter_direction));
  }
  else
  {
    return material_behaviour(false, vec3f(0.0));
  }
}

fn dielectric(normal : vec3f, r_direction: vec3f, refraction_index: f32, frontface: bool, random_sphere: vec3f, fuzz: f32, rng_state: ptr<function, u32>) -> material_behaviour
{  
  var refraction_ratio: f32;
  if (frontface) {
    refraction_ratio = 1.0 / refraction_index;
  } else {
    refraction_ratio = refraction_index;
  }

  var unit_direction = normalize(r_direction);

  var cos_theta = min(dot(-unit_direction, normal), 1.0);

  var sin_theta_sq = 1.0 - cos_theta * cos_theta;
  var sin_theta_2_sq = refraction_ratio * refraction_ratio * sin_theta_sq;
  
  var cannot_refract = sin_theta_2_sq > 1.0;
  var scatter_direction: vec3f;

  var r0 = (1.0 - refraction_ratio) / (1.0 + refraction_ratio);
  r0 = r0 * r0;
  var reflectance = r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);

  if (cannot_refract || reflectance > rng_next_float(rng_state))
  {
    scatter_direction = reflect(unit_direction, normal);
  }
  else
  {
    var perp = refraction_ratio * (unit_direction + cos_theta * normal);
    var parallel = -sqrt(abs(1.0 - sin_theta_2_sq)) * normal;
    scatter_direction = perp + parallel;
  }
  return material_behaviour(true, normalize(scatter_direction));
}

fn emmisive(color: vec3f, light: f32) -> material_behaviour
{
  return material_behaviour(false, color * light);
}

fn trace(r: ray, rng_state: ptr<function, u32>) -> vec3f
{
  var maxbounces = i32(uniforms[2]);

  var light = vec3f(0.0);

  var color = vec3f(1.0);

  var r_ = r;
  
  var backgroundcolor1 = int_to_rgb(i32(uniforms[11]));
  var backgroundcolor2 = int_to_rgb(i32(uniforms[12]));

  for (var j = 0; j < maxbounces; j = j + 1)
  {
    var rec = check_ray_collision(r_, RAY_TMAX);

    if (!rec.hit_anything)
    {
      light += color * envoriment_color(r_.direction, backgroundcolor1, backgroundcolor2);
      break;
    }

    var mat = rec.object_material;
    var mat_smoothness = mat.x; 
    var mat_absorption = mat.y;
    var mat_specular = mat.z;
    var mat_light = mat.w;

    var random_sphere = rng_next_vec3_in_unit_sphere(rng_state);
    var behaviour: material_behaviour;

    if (mat_light > 0.0)
    {
      behaviour = emmisive(rec.object_color.xyz, mat_light);
      light += color * behaviour.direction;
      break;
    }

    if (mat_smoothness > 0.0 && mat_specular > 0.5)
    {
      behaviour = metal(rec.normal, r_.direction, mat_absorption, random_sphere);
    }
    else if (mat_smoothness < 0.0)
    {
      behaviour = dielectric(rec.normal, r_.direction, mat_specular, rec.frontface, random_sphere, mat_absorption, rng_state);
    }
    else
    {
      behaviour = lambertian(rec.normal, mat_absorption, random_sphere, rng_state);
    }

    if (!behaviour.scatter)
    {
      break;
    }

    if (mat_smoothness > 0.0)
    {
      color *= mix(vec3f(1.0, 1.0, 1.0), rec.object_color.xyz, mat_specular);
    }
    else
    {
      color *= rec.object_color.xyz;
    }

    r_.origin = rec.p;
    r_.direction = behaviour.direction;
  }

  return light;
}

@compute @workgroup_size(THREAD_COUNT, THREAD_COUNT, 1)
fn render(@builtin(global_invocation_id) id : vec3u)
{
  var rez = uniforms[1];
  var time = u32(uniforms[0]);

  var rng_state = init_rng(vec2(id.x, id.y), vec2(u32(rez)), time);

  var lookfrom = vec3(uniforms[7], uniforms[8], uniforms[9]);
  var lookat = vec3(uniforms[23], uniforms[24], uniforms[25]);
  var cam = get_camera(lookfrom, lookat, vec3(0.0, 1.0, 0.0), uniforms[10], 1.0, uniforms[6], uniforms[5]);

  var color = vec3f(0.0);
  var samples_per_pixel = i32(uniforms[4]);
  var fragCoord = vec2f(f32(id.x), f32(id.y));

  for (var i = 0; i < samples_per_pixel; i = i + 1)
  {
    var uv = (fragCoord + sample_square(&rng_state)) / vec2(rez);

    var r = get_ray(cam, uv, &rng_state);

    color += trace(r, &rng_state);
  }

  var final_samples = max(1.0, f32(samples_per_pixel));
  color = color / final_samples;

  var color_out = vec4(linear_to_gamma(color), 1.0);
  var map_fb = mapfb(id.xy, rez);

  var should_accumulate = uniforms[3];

  // rtfb stores the SUM of all samples in xyz, and the COUNT in w.
  // We multiply by should_accumulate (0.0 or 1.0) to reset the sum if the camera moved.
  var accumulated_color = rtfb[map_fb] * should_accumulate;
  
  // Add the new color (r,g,b) and increment the sample count (w)
  accumulated_color += color_out; 

  // Store the new sum and count
  rtfb[map_fb] = accumulated_color;
  
  // The final display color is the SUM divided by the COUNT
  fb[map_fb] = accumulated_color / accumulated_color.w;
}