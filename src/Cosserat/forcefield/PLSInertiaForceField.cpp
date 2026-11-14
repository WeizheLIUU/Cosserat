/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
* Copyright (c) 2026. Distributed under the GNU LGPL, version 2.1 or later.     *
* See <https://www.gnu.org/licenses/>.                                         *
******************************************************************************/
#include <Cosserat/forcefield/PLSInertiaForceField.h>
#include <Cosserat/mapping/Strain2FramesCosseratMapping.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/accessor.h>
#include <cmath>

namespace Cosserat::forcefield
{
template<class Strain>
using Geometry = mapping::Strain2FramesCosseratMapping<Strain,
    sofa::defaulttype::Rigid3Types, sofa::defaulttype::Rigid3Types>;

PLSInertiaForceField::PLSInertiaForceField()
    : l_geometry(initLink("cosseratMapping", "PLS-enabled Strain2Frames mapping at the mass frames"))
    , l_rigidMapping(initLink("rigidMapping", "Existing RigidMapping from mass frames to these points"))
    , l_mass(initLink("mass", "Existing UniformMass/DiagonalMass<Vec3d> on exactly these points; no duplicate mass"))
    , d_enabled(initData(&d_enabled, true, "enabled", "Evaluate convective inertia bias"))
    , d_biasForces(initData(&d_biasForces, "biasForces", "Last Cartesian inertia correction, before native force transpose"))
    , d_motionBuilds(initData(&d_motionBuilds, 0u, "motionBuilds", "Shared geometry motion-cache rebuild count"))
{
    d_biasForces.setReadOnly(true);
    d_motionBuilds.setReadOnly(true);
}

bool PLSInertiaForceField::validate()
{
    using State = sofa::core::objectmodel::ComponentState;
    auto* rigid = l_rigidMapping.get();
    auto* mass = l_mass.get();
    auto* vec6 = dynamic_cast<Geometry<sofa::defaulttype::Vec6Types>*>(l_geometry.get());
    auto* vec3 = dynamic_cast<Geometry<sofa::defaulttype::Vec3Types>*>(l_geometry.get());
    const bool validGeometry = (vec6 && vec6->d_plsKinematics.getValue()
        && vec6->getFramesState() == (rigid ? rigid->getFromModel() : nullptr))
        || (vec3 && vec3->d_plsKinematics.getValue()
        && vec3->getFramesState() == (rigid ? rigid->getFromModel() : nullptr));
    const bool validMass = mass && (mass->getClassName() == "UniformMass" || mass->getClassName() == "DiagonalMass")
        && (mass->getTemplateName() == "Vec3d" || mass->getTemplateName() == "Vec3d,Vec3d")
        && mass->getMechanicalStates().size() == 1
        && mass->getMechanicalStates()[0] == this->mstate.get();
    if (!this->mstate || !rigid || rigid->getToModel() != this->mstate.get()
        || !validGeometry || !validMass || rigid->d_useX0.getValue()
        || rigid->d_indexFromEnd.getValue())
    {
        msg_error() << "Requires a PLS-enabled Strain2Frames mapping, its explicit-index/local-point "
                       "RigidMapping and a UniformMass/DiagonalMass<Vec3d> on the same Cartesian state.";
        this->d_componentState.setValue(State::Invalid);
        return false;
    }
    const auto& points = rigid->d_points.getValue();
    const auto& indices = rigid->d_rigidIndexPerPoint.getValue();
    if (points.size() != this->mstate->getSize() || indices.size() != points.size())
    {
        msg_error() << "Local points and rigidIndexPerPoint must match the mass point count.";
        this->d_componentState.setValue(State::Invalid);
        return false;
    }
    this->d_componentState.setValue(State::Valid);
    return true;
}

void PLSInertiaForceField::init()
{
    Inherit1::init();
    reinit();
}
void PLSInertiaForceField::reinit()
{
    validate();
}

void PLSInertiaForceField::addForce(const sofa::core::MechanicalParams* params,
    DataVecDeriv& dataForce, const DataVecCoord& dataX, const DataVecDeriv&)
{
    if (!validate())
        return;
    VecDeriv bias(dataX.getValue().size());
    if (d_enabled.getValue())
    {
        const std::vector<mapping::pls_detail::TransformJet>* jets = nullptr;
        std::size_t builds = 0;
        if (auto* geometry = dynamic_cast<Geometry<sofa::defaulttype::Vec6Types>*>(l_geometry.get()))
        {
            jets = geometry->getPLSFrameMotion(params);
            builds = geometry->getPLSMotionBuildCount();
        }
        else if (auto* geometry = dynamic_cast<Geometry<sofa::defaulttype::Vec3Types>*>(l_geometry.get()))
        {
            jets = geometry->getPLSFrameMotion(params);
            builds = geometry->getPLSMotionBuildCount();
        }
        if (!jets)
        {
            msg_error() << "Cannot evaluate finite PLS position/velocity motion jets.";
            return;
        }
        d_motionBuilds.setValue(static_cast<unsigned int>(builds));
        const auto& points = l_rigidMapping->d_points.getValue();
        const auto& indices = l_rigidMapping->d_rigidIndexPerPoint.getValue();
        for (std::size_t i = 0; i < bias.size(); ++i)
        {
            const auto index = indices[i];
            const auto mass = l_mass->getElementMass(i);
            if (index >= jets->size() || !std::isfinite(mass) || mass < 0)
            {
                msg_error() << "Invalid rigid frame index or Cartesian point mass.";
                return;
            }
            const Eigen::Vector4d local(points[i][0], points[i][1], points[i][2], 1.0);
            const Eigen::Vector4d acceleration = (*jets)[index].second * local;
            for (int c = 0; c < 3; ++c)
                bias[i][c] = -mass * acceleration[c];
            if (!acceleration.allFinite())
            {
                msg_error() << "Convective point acceleration overflowed.";
                return;
            }
        }
    }
    auto force = sofa::helper::getWriteAccessor(dataForce);
    force.resize(bias.size());
    for (std::size_t i = 0; i < bias.size(); ++i)
        force[i] += bias[i];
    d_biasForces.setValue(bias);
}
} // namespace Cosserat::forcefield

namespace Cosserat
{
void registerPLSInertiaForceField(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Convective inertia correction for existing PLS Cartesian mass quadrature; no duplicate mass.")
        .add<forcefield::PLSInertiaForceField>());
}
}
