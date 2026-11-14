/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
* Copyright (c) 2026. Distributed under the GNU LGPL, version 2.1 or later.     *
* See <https://www.gnu.org/licenses/>.                                         *
******************************************************************************/
#pragma once
#include <Cosserat/config.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/component/mapping/nonlinear/RigidMapping.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace Cosserat::forcefield
{
using PointInertiaBase = sofa::core::behavior::ForceField<sofa::defaulttype::Vec3Types>;
using PointRigidMapping = sofa::component::mapping::nonlinear::RigidMapping<
    sofa::defaulttype::Rigid3Types, sofa::defaulttype::Vec3Types>;

/// Add only -m Jdot*v at the existing Cartesian mass quadrature points.
/// Native mappings return -sum m J^T Jdot*v to independent DOFs; M stays native.
/// This bias is deliberately lagged/explicit, keeping the symmetric LDL system.
class SOFA_COSSERAT_API PLSInertiaForceField : public PointInertiaBase
{
public:
    SOFA_CLASS(PLSInertiaForceField, PointInertiaBase);
    static constexpr unsigned int flags = sofa::core::objectmodel::BaseLink::FLAG_STOREPATH
                                       | sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK;
    sofa::core::objectmodel::SingleLink<PLSInertiaForceField, sofa::core::BaseMapping, flags> l_geometry;
    sofa::core::objectmodel::SingleLink<PLSInertiaForceField, PointRigidMapping, flags> l_rigidMapping;
    sofa::core::objectmodel::SingleLink<PLSInertiaForceField, sofa::core::behavior::BaseMass, flags> l_mass;
    sofa::Data<bool> d_enabled;
    sofa::Data<VecDeriv> d_biasForces;
    sofa::Data<unsigned int> d_motionBuilds;

    void init() override;
    void reinit() override;
    void addForce(const sofa::core::MechanicalParams*, DataVecDeriv&, const DataVecCoord&, const DataVecDeriv&) override;
    // As documented by SOFA ForceField, omitted derivatives integrate this
    // nonconservative inertial bias explicitly. It is not viscous damping.
    void addDForce(const sofa::core::MechanicalParams*, DataVecDeriv&, const DataVecDeriv&) override {}
    void addKToMatrix(const sofa::core::MechanicalParams*, const sofa::core::behavior::MultiMatrixAccessor*) override {}
    void addBToMatrix(const sofa::core::MechanicalParams*, const sofa::core::behavior::MultiMatrixAccessor*) override {}
    void buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix*) override {}
    void buildDampingMatrix(sofa::core::behavior::DampingMatrix*) override {}
    double getPotentialEnergy(const sofa::core::MechanicalParams*, const DataVecCoord&) const override { return 0.0; }

protected:
    PLSInertiaForceField();
    ~PLSInertiaForceField() override = default;
private:
    bool validate();
};
} // namespace Cosserat::forcefield
