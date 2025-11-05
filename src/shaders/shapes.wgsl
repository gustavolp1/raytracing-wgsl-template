fn hit_sphere(center: vec3f, radius: f32, r: ray, record: ptr<function, hit_record>, max: f32)
{
  // This function solves the quadratic equation for ray-sphere intersection.
  // A ray is P(t) = A + t*B (where A=r.origin, B=r.direction)
  // A sphere is (P - C) * (P - C) = radius*radius (where C=center)
  //
  // Substitute P(t) into the sphere equation:
  // (A + t*B - C) * (A + t*B - C) = radius*radius
  //
  // Let oc = A - C (vector from ray origin to sphere center)
  // (t*B + oc) * (t*B + oc) = radius*radius
  //
  // Expand the dot product:
  // t*t*(B*B) + 2*t*(B*oc) + (oc*oc) - radius*radius = 0
  //
  // This is a quadratic equation at^2 + bt + c = 0
  // a = dot(B, B)
  // b = 2 * dot(B, oc)
  // c = dot(oc, oc) - radius*radius

  let oc = r.origin - center;
  let a = dot(r.direction, r.direction);
  let half_b = dot(oc, r.direction); // We use half_b for a simpler discriminant calculation
  let c = dot(oc, oc) - radius * radius;

  // Discriminant = (2*half_b)^2 - 4*a*c = 4*(half_b*half_b - a*c)
  let discriminant = half_b * half_b - a * c;

  // If discriminant is negative, the ray misses the sphere.
  if (discriminant < 0.0)
  {
    record.hit_anything = false;
    return;
  }

  let sqrtd = sqrt(discriminant);

  // Find the nearest root (smallest t) that is in the acceptable range
  var root = (-half_b - sqrtd) / a;

  // Check the first root (t0)
  if (root < RAY_TMIN || root > max)
  {
    // First root is not valid, check the second root (t1)
    root = (-half_b + sqrtd) / a;
    if (root < RAY_TMIN || root > max)
    {
      // Neither root is valid, no hit.
      record.hit_anything = false;
      return;
    }
  }

  // We have a valid hit! Record the intersection details.
  record.t = root;
  record.p = ray_at(r, record.t);
  let outward_normal = (record.p - center) / radius;
  
  // Check if the ray hit the front or back of the surface
  record.frontface = dot(r.direction, outward_normal) < 0.0;
  // Make the normal always point against the incoming ray
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