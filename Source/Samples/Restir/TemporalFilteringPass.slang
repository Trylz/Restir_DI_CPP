#include "Light.slangh"
#include "Reservoir.slangh"

import Utils.Sampling.TinyUniformSampleGenerator;

cbuffer PerFrameCB
{
    uint2 viewportDims;
    float3 cameraPositionWs;
    float4x4 previousFrameViewProjMat;
    uint nbReservoirPerPixel;
    uint sampleIndex;
    uint motion;
    float temporalLinearDepthThreshold;
    float temporalWsRadiusThreshold;
    float temporalNormalThreshold;
};

RWStructuredBuffer<RestirReservoir> gCurrentFrameReservoirs;
StructuredBuffer<RestirReservoir> gPreviousFrameReservoirs;

Texture2D<float4> gCurrentPositionWs;
Texture2D<float4> gPreviousPositionWs;

Texture2D<float4> gCurrentNormalWs;
Texture2D<float4> gPreviousNormalWs;

Texture2D<float4> gAlbedo;
Texture2D<float4> gSpecular;


[shader("closesthit")]
void primaryClosestHit(inout PrimaryRayData hitData, BuiltInTriangleIntersectionAttributes attribs)
{
}

[shader("anyhit")]
void primaryAnyHit(inout PrimaryRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
}

[shader("miss")]
void primaryMiss(inout PrimaryRayData hitData)
{
}

[shader("miss")]
void shadowMiss(inout ShadowRayData hitData)
{
    hitData.hit = false;
}

[shader("anyhit")]
void shadowAnyHit(inout ShadowRayData hitData, BuiltInTriangleIntersectionAttributes attribs)
{
    hitData.hit = true;
}

int2 getPreviousFramePixelPos(float4 P, float width, float height)
{
    float4 ndc = mul(previousFrameViewProjMat, P);
    ndc = ndc / ndc.w;
    float2 s = (ndc.xy + float2(1, 1)) * 0.5f;
    s = float2(s.x, 1.f - s.y);
    return (int2)(s * float2(width, height));
}

[shader("raygeneration")] void rayGen()
{
    const uint2 pixel = threadId.xy;
    if (any(pixel >= viewportDims))
        return;

    if (gCurrentPositionWs[pixel].w == 0.0f)
        return;

    const float3 currP = gCurrentPositionWs[pixel].xyz;

    int2 previousPixelPos = getPreviousFramePixelPos(float4(currP, 1.0f), (float)viewportDims.x, (float)viewportDims.y);
    if (previousPixelPos.x < 0 || previousPixelPos.x >= (int)viewportDims.x)
        return;
    if (previousPixelPos.y < 0 || previousPixelPos.y >= (int)viewportDims.y)
        return;
    if (gPreviousPositionWs[previousPixelPos].w == 0.0f)
        return;

    const float3 prevP = gPreviousPositionWs[previousPixelPos].xyz;
    if (length(prevP - currP) > temporalWsRadiusThreshold)
        return;

    const float3 currN = gCurrentNormalWs[pixel].xyz;
    const float3 prevN = gPreviousNormalWs[previousPixelPos].xyz;

    if (length(prevN - currN) > temporalNormalThreshold)
        return;

    const float currlinearDepth = gCurrentNormalWs[pixel].w;
    const float prevlinearDepth = gPreviousNormalWs[previousPixelPos].w;

    if (abs(currlinearDepth - prevlinearDepth) > temporalLinearDepthThreshold)
        return;

    const float3 V = normalize(cameraPositionWs - currP);
    const float3 diffuse = gAlbedo[pixel].xyz;
    const float3 specular = gSpecular[pixel].xyz;
    const float roughness = gSpecular[pixel].w;

    // Current pixel
    const uint currentPixelLinearIndex = pixel.y * viewportDims.x + pixel.x;
    const uint currentPixelReservoirsStart = currentPixelLinearIndex * nbReservoirPerPixel;

    // Previous pixel
    const uint previousPixelLinearIndex = previousPixelPos.y * viewportDims.x + previousPixelPos.x;
    const uint previousPixelReservoirsStart = previousPixelLinearIndex * nbReservoirPerPixel;

    // Init rng
    TinyUniformSampleGenerator rng = TinyUniformSampleGenerator(pixel.xy, sampleIndex);

    // Combine reservoirs
    for (uint i = 0; i < nbReservoirPerPixel; ++i)
    {
        RestirReservoir currentReservoir = gCurrentFrameReservoirs[currentPixelReservoirsStart + i];
        RestirReservoir previousReservoir = gPreviousFrameReservoirs[previousPixelReservoirsStart + i];
        if (motion > 0)
        {
            // Clamp M according to paper.
            previousReservoir.mM = min(20 * currentReservoir.mM, previousReservoir.mM);

            // Check visibility: Unbiased Restir.
            float3 L = r.mY.mLightSamplePosition - currP;
            const float Llen = length(L);
            L /= Llen;

            RayDesc ray;
            ray.Origin = currP;
            ray.Direction = L;
            ray.TMin = 0.001f;
            ray.TMax = Llen;

            ShadowRayData rayData;
            rayData.hit = true;
            TraceRay(gScene.rtAccel, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, 1 /* ray index */, rayTypeCount, 1, ray, rayData);

            if (rayData.hit)
            {
                previousReservoir.m_W = 0.0f;
            }
        }

        gCurrentFrameReservoirs[currentPixelReservoirsStart + i] =
            combineReservoirs(currentReservoir, previousReservoir, currP, currN, V, diffuse, specular, roughness, rng);
    }
}








