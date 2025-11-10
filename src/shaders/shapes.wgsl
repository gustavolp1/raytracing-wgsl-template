fn hit_sphere(center: vec3f, radius: f32, r: ray, record: ptr<function, hit_record>, max: f32)
{
  let oc = r.origin - center;
  let a = dot(r.direction, r.direction);
  let half_b = dot(oc, r.direction);
  let c = dot(oc, oc) - radius * radius;

  let discriminant = half_b * half_b - a * c;

  if (discriminant < 0.0)
  {
    record.hit_anything = false;
    return;
  }

  let sqrtd = sqrt(discriminant);

  var root = (-half_b - sqrtd) / a;

  if (root < RAY_TMIN || root > max)
  {
    root = (-half_b + sqrtd) / a;
    if (root < RAY_TMIN || root > max)
    {
      record.hit_anything = false;
      return;
    }
  }

  record.t = root;
  record.p = ray_at(r, record.t);
  let outward_normal = (record.p - center) / radius;

  record.frontface = dot(r.direction, outward_normal) < 0.0;
  record.normal = select(-outward_normal, outward_normal, record.frontface);
  
  record.hit_anything = true;
}

fn hit_quad(r: ray, Q: vec4f, u: vec4f, v: vec4f, record: ptr<function, hit_record>, max: f32)
{
  var n = cross(u.xyz, v.xyz);
  var normal = normalize(n);
  var D = dot(normal, Q.xyz);
  var w = n / dot(n.xyz, n.xyz);

  var denom = dot(normal, r.direction);
  if (abs(denom) < 0.0001)
  {
    record.hit_anything = false;
    return;
  }

  var t = (D - dot(normal, r.origin)) / denom;
  if (t < RAY_TMIN || t > max)
  {
    record.hit_anything = false;
    return;
  }

  var intersection = ray_at(r, t);
  var planar_hitpt_vector = intersection - Q.xyz;
  var alpha = dot(w, cross(planar_hitpt_vector, v.xyz));
  var beta = dot(w, cross(u.xyz, planar_hitpt_vector));

  if (alpha < 0.0 || alpha > 1.0 || beta < 0.0 || beta > 1.0)
  {
    record.hit_anything = false;
    return;
  }

  if (dot(normal, r.direction) > 0.0)
  {
    record.hit_anything = false;
    return;
  }

  record.t = t;
  record.p = intersection;
  record.normal = normal;
  record.hit_anything = true;
}

fn hit_triangle(r: ray, v0: vec3f, v1: vec3f, v2: vec3f, record: ptr<function, hit_record>, max: f32)
{
  var v1v0 = v1 - v0;
  var v2v0 = v2 - v0;
  var rov0 = r.origin - v0;

  var n = cross(v1v0, v2v0);
  var q = cross(rov0, r.direction);

  var d = 1.0 / dot(r.direction, n);

  var u = d * dot(-q, v2v0);
  var v = d * dot(q, v1v0);
  var t = d * dot(-n, rov0);

  if (u < 0.0 || u > 1.0 || v < 0.0 || (u + v) > 1.0)
  {
    record.hit_anything = false;
    return;
  }

  if (t < RAY_TMIN || t > max)
  {
    record.hit_anything = false;
    return;
  }

  record.t = t;
  record.p = ray_at(r, t);
  record.normal = normalize(n);
  record.hit_anything = true;
}

fn hit_box(r: ray, center: vec3f, rad: vec3f, record: ptr<function, hit_record>, t_max: f32)
{
  var m = 1.0 / r.direction;
  var n = m * (r.origin - center);
  var k = abs(m) * rad;

  var t1 = -n - k;
  var t2 = -n + k;

  var tN = max(max(t1.x, t1.y), t1.z);
  var tF = min(min(t2.x, t2.y), t2.z);

  if (tN > tF || tF < 0.0)
  {
    record.hit_anything = false;
    return;
  }

  var t = tN;
  if (t < RAY_TMIN || t > t_max)
  {
    record.hit_anything = false;
    return;
  }

  record.t = t;
  record.p = ray_at(r, t);
  record.normal = -sign(r.direction) * step(t1.yzx, t1.xyz) * step(t1.zxy, t1.xyz);
  record.hit_anything = true;

  return;
}

fn hit_cylinder(r: ray, base_center: vec3f, axis: vec3f, cyl_radius: f32, cyl_height: f32, record: ptr<function, hit_record>, t_max: f32)
{
  const EPSILON: f32 = 1e-6;

  let ray_origin = r.origin;
  let ray_dir = r.direction;
  let origin_to_base = ray_origin - base_center;

  let dir_dot_axis = dot(ray_dir, axis);
  let oc_dot_axis = dot(origin_to_base, axis);

  let quad_a = dot(ray_dir, ray_dir) - dir_dot_axis * dir_dot_axis;
  let quad_half_b = dot(ray_dir, origin_to_base) - dir_dot_axis * oc_dot_axis;
  let quad_c = dot(origin_to_base, origin_to_base) - oc_dot_axis * oc_dot_axis - cyl_radius * cyl_radius;

  var t_side = t_max + EPSILON;
  var hit_side = false;
  var normal_side = vec3f(0.0);
  var point_side = vec3f(0.0);

  if (abs(quad_a) > EPSILON) {
    let discriminant = quad_half_b * quad_half_b - quad_a * quad_c;
    if (discriminant >= 0.0) {
      let sqrt_disc = sqrt(discriminant);

      var t0 = (-quad_half_b - sqrt_disc) / quad_a;
      if (t0 >= RAY_TMIN && t0 <= t_max) {
        let m0 = dir_dot_axis * t0 + oc_dot_axis;
        if (m0 >= 0.0 && m0 <= cyl_height) {
          hit_side = true;
          t_side = t0;
        }
      }

      if (!hit_side) {
        var t1 = (-quad_half_b + sqrt_disc) / quad_a;
        if (t1 >= RAY_TMIN && t1 <= t_max) {
          let m1 = dir_dot_axis * t1 + oc_dot_axis;
          if (m1 >= 0.0 && m1 <= cyl_height) {
            hit_side = true;
            t_side = t1;
          }
        }
      }

      if (hit_side) {
        point_side = ray_at(r, t_side);
        let m = dir_dot_axis * t_side + oc_dot_axis;
        normal_side = normalize(point_side - (base_center + axis * m));
      }
    }
  }

  var t_cap = t_max + EPSILON;
  var hit_cap = false;
  var normal_cap = vec3f(0.0);
  var point_cap = vec3f(0.0);

  let normal_base = -axis;
  let denom_base = dot(normal_base, ray_dir);
  if (abs(denom_base) > EPSILON) {
    let t_base_plane = dot(normal_base, (base_center - ray_origin)) / denom_base;
    if (t_base_plane >= RAY_TMIN && t_base_plane <= t_max) {
      let p_base = ray_at(r, t_base_plane);
      if (length(p_base - base_center) <= cyl_radius + EPSILON) {
        hit_cap = true;
        t_cap = t_base_plane;
        normal_cap = normal_base;
        point_cap = p_base;
      }
    }
  }

  let end_center = base_center + axis * cyl_height;
  let normal_end = axis;
  let denom_end = dot(normal_end, ray_dir);
  if (abs(denom_end) > EPSILON) {
    let t_end_plane = dot(normal_end, (end_center - ray_origin)) / denom_end;
    if (t_end_plane >= RAY_TMIN && t_end_plane <= t_max && t_end_plane < t_cap) {
      let p_end = ray_at(r, t_end_plane);
      if (length(p_end - end_center) <= cyl_radius + EPSILON) {
        hit_cap = true;
        t_cap = t_end_plane;
        normal_cap = normal_end;
        point_cap = p_end;
      }
    }
  }

  if (!hit_side && !hit_cap) {
    record.hit_anything = false;
    return;
  }

  if (t_side < t_cap) {
    record.t = t_side;
    record.p = point_side;
    record.normal = normal_side;
  } else {
    record.t = t_cap;
    record.p = point_cap;
    record.normal = normal_cap;
  }
  
  record.hit_anything = true;
}