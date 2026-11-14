/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
* Copyright (c) 2026. Distributed under the GNU LGPL, version 2.1 or later.     *
* See <https://www.gnu.org/licenses/>.                                         *
******************************************************************************/
#pragma once

#include <Cosserat/config.h>
#include <sofa/core/Mapping.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>

namespace Cosserat::mapping
{

/**
 * @brief Evaluate a continuous piecewise linear strain field at material abscissas.
 *
 * Independent DOFs are nodal strains. Samples are dependent mechanical states,
 * suitable for the existing Cosserat force fields and PCS geometry mappings.
 * This component performs interpolation only; spatial integration belongs to
 * the downstream components. Vec6 follows Cosserat's [curvature, extension/shear]
 * convention without adding the reference axial strain a second time.
 */
template<class DataTypes>
class PiecewiseLinearStrainMapping : public sofa::core::Mapping<DataTypes, DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PiecewiseLinearStrainMapping, DataTypes),
               SOFA_TEMPLATE2(sofa::core::Mapping, DataTypes, DataTypes));

    using Inherit = sofa::core::Mapping<DataTypes, DataTypes>;
    using Real = typename DataTypes::Real;
    using VecCoord = typename DataTypes::VecCoord;
    using VecDeriv = typename DataTypes::VecDeriv;
    using MatrixDeriv = typename DataTypes::MatrixDeriv;
    using Abscissas = sofa::type::vector<Real>;
    using Jacobians = sofa::type::vector<sofa::linearalgebra::BaseMatrix*>;

    sofa::Data<Abscissas> d_nodeAbscissa; ///< Strictly increasing material coordinates of input strain nodes.
    sofa::Data<Abscissas> d_sampleAbscissa; ///< Material coordinates of output samples, within the nodal domain.

    void init() override;
    void reinit() override;
    bool isLinear() const override;
    void apply(const sofa::core::MechanicalParams*, sofa::Data<VecCoord>&,
               const sofa::Data<VecCoord>&) override;
    void applyJ(const sofa::core::MechanicalParams*, sofa::Data<VecDeriv>&,
                const sofa::Data<VecDeriv>&) override;
    void applyJT(const sofa::core::MechanicalParams*, sofa::Data<VecDeriv>&,
                 const sofa::Data<VecDeriv>&) override;
    void applyJT(const sofa::core::ConstraintParams*, sofa::Data<MatrixDeriv>&,
                 const sofa::Data<MatrixDeriv>&) override;
    void computeAccFromMapping(const sofa::core::MechanicalParams*, sofa::Data<VecDeriv>&,
                              const sofa::Data<VecDeriv>&, const sofa::Data<VecDeriv>&) override;
    const sofa::linearalgebra::BaseMatrix* getJ() override;
    const sofa::linearalgebra::BaseMatrix* getJ(const sofa::core::MechanicalParams*) override;
    const Jacobians* getJs() override;

protected:
    PiecewiseLinearStrainMapping();
    ~PiecewiseLinearStrainMapping() override = default;

private:
    bool ensureReady();
    bool hasInputSize(std::size_t size);
    bool hasSampleSize(std::size_t size);
    void interpolate(sofa::Data<VecCoord>& out, const sofa::Data<VecCoord>& in);

    Abscissas m_cachedNodes;
    Abscissas m_cachedSamples;
    sofa::type::vector<sofa::Index> m_leftNodes;
    Abscissas m_rightWeights;
    sofa::linearalgebra::CompressedRowSparseMatrix<Real> m_jacobian;
    Jacobians m_jacobians;
    std::size_t m_cachedInputSize = 0;
    bool m_hasCache = false;
};

extern template class SOFA_COSSERAT_API PiecewiseLinearStrainMapping<sofa::defaulttype::Vec3Types>;
extern template class SOFA_COSSERAT_API PiecewiseLinearStrainMapping<sofa::defaulttype::Vec6Types>;

} // namespace Cosserat::mapping
