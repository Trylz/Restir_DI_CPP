#include "TemporalFilteringPass.h"
#include "GBuffer.h"
#include "ReservoirManager.h"
#include "SceneSettings.h"

namespace Restir
{
using namespace Falcor;

TemporalFilteringPass::TemporalFilteringPass(
    ref<Device> pDevice,
    Falcor::ref<Falcor::Scene> pScene,
    uint32_t width,
    uint32_t height
)
    : mpScene(pScene), mWidth(width), mHeight(height)
{
    compileProgram(pDevice);
}

void TemporalFilteringPass::compileProgram(Falcor::ref<Falcor::Device> pDevice)
{
    auto shaderModules = mpScene->getShaderModules();
    auto typeConformances = mpScene->getTypeConformances();

    auto defines = mpScene->getSceneDefines();

    ProgramDesc rtProgDesc;
    rtProgDesc.addShaderModules(shaderModules);
    rtProgDesc.addShaderLibrary("Samples/Restir/TemporalFilteringPass.slang");
    rtProgDesc.addTypeConformances(typeConformances);
    rtProgDesc.setMaxTraceRecursionDepth(1);

    rtProgDesc.setMaxPayloadSize(24);

    ref<RtBindingTable> sbt = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
    sbt->setRayGen(rtProgDesc.addRayGen("rayGen"));
    sbt->setMiss(0, rtProgDesc.addMiss("primaryMiss"));
    sbt->setMiss(1, rtProgDesc.addMiss("shadowMiss"));
    auto primary = rtProgDesc.addHitGroup("primaryClosestHit", "primaryAnyHit");
    auto shadow = rtProgDesc.addHitGroup("", "shadowAnyHit");
    sbt->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), primary);
    sbt->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), shadow);

    mpTemporalFilteringPass = Program::create(pDevice, rtProgDesc, defines);
    mpRtVars = RtProgramVars::create(pDevice, mpTemporalFilteringPass, sbt);
}

void TemporalFilteringPass::render(Falcor::RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "TemporalFilteringPass::render");

    auto var = mpRtVars->getRootVar();

    var["PerFrameCB"]["viewportDims"] = uint2(mWidth, mHeight);
    var["PerFrameCB"]["cameraPositionWs"] = mpScene->getCamera()->getPosition();
    var["PerFrameCB"]["previousFrameViewProjMat"] =  mPreviousFrameViewProjMat;
    var["PerFrameCB"]["nbReservoirPerPixel"] = SceneSettingsSingleton::instance()->nbReservoirPerPixel;
    var["PerFrameCB"]["sampleIndex"] = ++mSampleIndex;
    var["PerFrameCB"]["motion"] = (uint)(mPreviousFrameViewProjMat != mpScene->getCamera()->getViewProjMatrix());

    var["PerFrameCB"]["temporalLinearDepthThreshold"] = SceneSettingsSingleton::instance()->temporalLinearDepthThreshold;
    var["PerFrameCB"]["temporalWsRadiusThreshold"] = SceneSettingsSingleton::instance()->temporalWsRadiusThreshold;
    var["PerFrameCB"]["temporalNormalThreshold"] = SceneSettingsSingleton::instance()->temporalNormalThreshold;

    var["gCurrentFrameReservoirs"] = ReservoirManagerSingleton::instance()->getCurrentFrameReservoirBuffer();
    var["gPreviousFrameReservoirs"] = ReservoirManagerSingleton::instance()->getPreviousFrameReservoirBuffer();

    var["gCurrentPositionWs"] = GBufferSingleton::instance()->getCurrentPositionWsTexture();
    var["gPreviousPositionWs"] = GBufferSingleton::instance()->getPreviousPositionWsTexture();

    var["gCurrentNormalWs"] = GBufferSingleton::instance()->getCurrentNormalWsTexture();
    var["gPreviousNormalWs"] = GBufferSingleton::instance()->getPreviousNormalWsTexture();
    var["gAlbedo"] = GBufferSingleton::instance()->getAlbedoTexture();
    var["gSpecular"] = GBufferSingleton::instance()->getSpecularTexture();

    mpScene->raytrace(pRenderContext, mpTemporalFilteringPass.get(), mpRtVars, uint3(mWidth, mHeight, 1));
    mPreviousFrameViewProjMat = mpScene->getCamera()->getViewProjMatrix();
}
} // namespace Restir
