#include <optix.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_math_namespace.h>

#include "optix/vertex_attr.h"

rtBuffer<VertexAttr> attributesBuffer;
rtBuffer<uint3>            indicesBuffer;

// Axis Aligned Bounding Box routine for indexed interleaved triangle data.
RT_PROGRAM void bounds(int primitiveIndex, float result[6])
{
  const uint3 indices = indicesBuffer[primitiveIndex];

  const float3 v0 = attributesBuffer[indices.x].vertex;
  const float3 v1 = attributesBuffer[indices.y].vertex;
  const float3 v2 = attributesBuffer[indices.z].vertex;

  const float area = optix::length(optix::cross(v1 - v0, v2 - v0));

  optix::Aabb *aabb = (optix::Aabb *) result;

  if (0.0f < area && !isinf(area))
  {
    aabb->m_min = fminf(fminf(v0, v1), v2);
    aabb->m_max = fmaxf(fmaxf(v0, v1), v2);
  }
  else
  {
    aabb->invalidate();
  }
}

// Attributes.
rtDeclareVariable(optix::float3, varNormal,    attribute NORMAL, ); 

rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );

// Intersection routine for indexed interleaved triangle data.
RT_PROGRAM void intersect(int primitiveIndex)
{
  const uint3 indices = indicesBuffer[primitiveIndex];

  VertexAttr const& a0 = attributesBuffer[indices.x];
  VertexAttr const& a1 = attributesBuffer[indices.y];
  VertexAttr const& a2 = attributesBuffer[indices.z];

  float3 n;
  float  t;
  float  beta;
  float  gamma;

  if (intersect_triangle(theRay, a0.vertex, a1.vertex, a2.vertex, n, t, beta, gamma))
  {
    if (rtPotentialIntersection(t))
    {
      // Barycentric interpolation:
      const float alpha = 1.0f - beta - gamma;

      // Note: No normalization on the TBN attributes here for performance reasons.
      //       It's done after the transformation into world space anyway.
      varNormal         = a0.normal   * alpha + a1.normal   * beta + a2.normal   * gamma;
      
      rtReportIntersection(0);
    }
  }
}
