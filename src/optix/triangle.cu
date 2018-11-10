#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(float3, v1, , );
rtDeclareVariable(float3, v2, , );
rtDeclareVariable(float3, v3, , );
rtDeclareVariable(float3, normal, , );
rtDeclareVariable(float3, v1_normal, , );
rtDeclareVariable(float3, v2_normal, , );
rtDeclareVariable(float3, v3_normal, , );

//rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
//rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void intersect(int primIdx)
{
  float t, beta, gamma;
  if (intersect_triangle_branchless(ray,
      v1, v2, v3, normal, t, beta, gamma)) {
    if (rtPotentialIntersection( t )) {
      shading_normal = v1_normal * beta + v2_normal * gamma + v3_normal * (1 - beta - gamma);
      rtReportIntersection( 0 );
    }
  }
}

RT_PROGRAM void bounds (int, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->m_min = fminf( fminf( v1, v2 ), v3 );
  aabb->m_max = fmaxf( fmaxf( v1, v2 ), v3 );
}
