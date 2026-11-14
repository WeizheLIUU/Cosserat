/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
* Copyright (c) 2026. Distributed under the GNU LGPL, version 2.1 or later.     *
* See <https://www.gnu.org/licenses/>.                                         *
******************************************************************************/
#include <Cosserat/mapping/PiecewiseLinearStrainMapping.h>

#include <sofa/core/Mapping.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/accessor.h>
#include <algorithm>
#include <cmath>

namespace Cosserat::mapping
{

template<class DataTypes>
PiecewiseLinearStrainMapping<DataTypes>::PiecewiseLinearStrainMapping()
    : d_nodeAbscissa(this->initData(&d_nodeAbscissa, "nodeAbscissa",
                                  "Strictly increasing material abscissas; one per independent strain node"))
    , d_sampleAbscissa(this->initData(&d_sampleAbscissa, "sampleAbscissa",
                                    "Material abscissas of dependent strain samples; extrapolation is forbidden"))
    , m_jacobians{&m_jacobian}
{
    this->f_applyRestPosition.setValue(true);
    this->f_mapMatrices.setValue(true);
}

template<class DataTypes>
bool PiecewiseLinearStrainMapping<DataTypes>::isLinear() const
{
    return true;
}

template<class DataTypes>
void PiecewiseLinearStrainMapping<DataTypes>::init()
{
    reinit();
    if (this->isComponentStateValid())
    {
        Inherit::init();
    }
}

template<class DataTypes>
void PiecewiseLinearStrainMapping<DataTypes>::reinit()
{
    using sofa::core::objectmodel::ComponentState;
    m_hasCache = true;
    m_cachedNodes = d_nodeAbscissa.getValue();
    m_cachedSamples = d_sampleAbscissa.getValue();
    m_cachedInputSize = this->getFromModel() ? this->getFromModel()->getSize() : 0;
    m_leftNodes.clear();
    m_rightWeights.clear();
    m_jacobian.clear();
    m_jacobian.resize(0, 0);
    this->d_componentState.setValue(ComponentState::Invalid);

    if (!this->getFromModel() || !this->getToModel())
    {
        msg_error() << "Both input and output mechanical states must be linked.";
        return;
    }
    if (this->getFromModel() == this->getToModel())
    {
        msg_error() << "Input and output must be different mechanical states.";
        return;
    }
    if (m_cachedNodes.size() < 2 || m_cachedNodes.size() != m_cachedInputSize)
    {
        msg_error() << "nodeAbscissa needs at least two entries and one entry per input strain node.";
        return;
    }
    for (std::size_t i = 0; i < m_cachedNodes.size(); ++i)
    {
        if (!std::isfinite(m_cachedNodes[i])
            || (i > 0 && (!(m_cachedNodes[i] > m_cachedNodes[i - 1])
                          || !std::isfinite(m_cachedNodes[i] - m_cachedNodes[i - 1]))))
        {
            msg_error() << "nodeAbscissa must be finite and strictly increasing.";
            return;
        }
    }
    for (const Real sample : m_cachedSamples)
    {
        if (!std::isfinite(sample) || sample < m_cachedNodes.front() || sample > m_cachedNodes.back())
        {
            msg_error() << "Every sampleAbscissa must be finite and inside the nodal domain.";
            return;
        }
    }

    constexpr auto dimension = DataTypes::deriv_total_size;
    m_jacobian.resize(m_cachedSamples.size() * dimension, m_cachedNodes.size() * dimension);
    for (std::size_t i = 0; i < m_cachedSamples.size(); ++i)
    {
        const Real sample = m_cachedSamples[i];
        const auto upper = std::upper_bound(m_cachedNodes.begin(), m_cachedNodes.end(), sample);
        const auto left = static_cast<sofa::Index>(
            std::min<std::size_t>(upper - m_cachedNodes.begin() - 1, m_cachedNodes.size() - 2));
        const Real weight = (sample - m_cachedNodes[left]) / (m_cachedNodes[left + 1] - m_cachedNodes[left]);
        m_leftNodes.push_back(left);
        m_rightWeights.push_back(weight);
        for (sofa::Size axis = 0; axis < dimension; ++axis)
        {
            m_jacobian.set(i * dimension + axis, left * dimension + axis, Real(1) - weight);
            m_jacobian.set(i * dimension + axis, (left + 1) * dimension + axis, weight);
        }
    }
    m_jacobian.compress();
    this->getToModel()->resize(m_cachedSamples.size());
    this->d_componentState.setValue(ComponentState::Valid);
    if (this->f_applyRestPosition.getValue())
    {
        const auto* rest = this->getFromModel()->read(sofa::core::vec_id::read_access::restPosition);
        if (rest->getValue().size() == m_cachedNodes.size())
        {
            interpolate(*this->getToModel()->write(sofa::core::vec_id::write_access::restPosition), *rest);
        }
    }
}

template<class DataTypes>
bool PiecewiseLinearStrainMapping<DataTypes>::ensureReady()
{
    const auto inputSize = this->getFromModel() ? this->getFromModel()->getSize() : 0;
    if (!m_hasCache || m_cachedNodes != d_nodeAbscissa.getValue()
        || m_cachedSamples != d_sampleAbscissa.getValue() || m_cachedInputSize != inputSize)
    {
        reinit();
    }
    return this->isComponentStateValid();
}

template<class DataTypes>
bool PiecewiseLinearStrainMapping<DataTypes>::hasInputSize(const std::size_t size)
{
    if (size == m_cachedNodes.size())
    {
        return true;
    }
    msg_error() << "Input vector size does not match nodeAbscissa.";
    return false;
}

template<class DataTypes>
bool PiecewiseLinearStrainMapping<DataTypes>::hasSampleSize(const std::size_t size)
{
    if (size == m_cachedSamples.size())
    {
        return true;
    }
    msg_error() << "Child vector size does not match sampleAbscissa.";
    return false;
}

template<class DataTypes>
void PiecewiseLinearStrainMapping<DataTypes>::interpolate(sofa::Data<VecCoord>& dOut,
                                                       const sofa::Data<VecCoord>& dIn)
{
    if (!ensureReady() || !hasInputSize(dIn.getValue().size()))
    {
        return;
    }
    const auto in = sofa::helper::getReadAccessor(dIn);
    auto out = sofa::helper::getWriteOnlyAccessor(dOut);
    out.resize(m_leftNodes.size());
    for (std::size_t i = 0; i < m_leftNodes.size(); ++i)
    {
        const auto left = m_leftNodes[i];
        const Real weight = m_rightWeights[i];
        out[i] = in[left] * (Real(1) - weight) + in[left + 1] * weight;
    }
}

template<class DataTypes>
void PiecewiseLinearStrainMapping<DataTypes>::apply(const sofa::core::MechanicalParams*,
    sofa::Data<VecCoord>& out, const sofa::Data<VecCoord>& in)
{
    interpolate(out, in);
}

template<class DataTypes>
void PiecewiseLinearStrainMapping<DataTypes>::applyJ(const sofa::core::MechanicalParams*,
    sofa::Data<VecDeriv>& out, const sofa::Data<VecDeriv>& in)
{
    interpolate(out, in);
}

template<class DataTypes>
void PiecewiseLinearStrainMapping<DataTypes>::computeAccFromMapping(const sofa::core::MechanicalParams* params,
    sofa::Data<VecDeriv>& out, const sofa::Data<VecDeriv>&, const sofa::Data<VecDeriv>& acceleration)
{
    applyJ(params, out, acceleration);
}

template<class DataTypes>
void PiecewiseLinearStrainMapping<DataTypes>::applyJT(const sofa::core::MechanicalParams*,
    sofa::Data<VecDeriv>& dOut, const sofa::Data<VecDeriv>& dIn)
{
    if (!ensureReady() || !hasSampleSize(dIn.getValue().size()))
    {
        return;
    }
    const auto in = sofa::helper::getReadAccessor(dIn);
    auto out = sofa::helper::getWriteAccessor(dOut);
    out.resize(m_cachedNodes.size());
    for (std::size_t i = 0; i < m_leftNodes.size(); ++i)
    {
        const auto left = m_leftNodes[i];
        const Real weight = m_rightWeights[i];
        out[left] += in[i] * (Real(1) - weight);
        out[left + 1] += in[i] * weight;
    }
}

template<class DataTypes>
void PiecewiseLinearStrainMapping<DataTypes>::applyJT(const sofa::core::ConstraintParams*,
    sofa::Data<MatrixDeriv>& dOut, const sofa::Data<MatrixDeriv>& dIn)
{
    if (!ensureReady())
    {
        return;
    }
    const auto& in = dIn.getValue();
    for (auto row = in.begin(); row != in.end(); ++row)
    {
        for (auto column = row.begin(); column != row.end(); ++column)
        {
            if (column.index() < 0 || static_cast<std::size_t>(column.index()) >= m_leftNodes.size())
            {
                msg_error() << "Constraint column index exceeds the sample count.";
                return;
            }
        }
    }
    auto out = sofa::helper::getWriteAccessor(dOut);
    // Many integration samples share a few independent PLS nodes. Reduce
    // their contributions before emitting sparse blocks, as in J_red=J*Tran.
    // addCol still accumulates into any pre-existing destination row.
    VecDeriv reduced(m_cachedInputSize);
    for (auto row = in.begin(); row != in.end(); ++row)
    {
        std::fill(reduced.begin(), reduced.end(), typename VecDeriv::value_type{});
        for (auto column = row.begin(); column != row.end(); ++column)
        {
            const auto sample = static_cast<std::size_t>(column.index());
            const auto left = m_leftNodes[sample];
            const Real weight = m_rightWeights[sample];
            reduced[left] += column.val() * (Real(1) - weight);
            reduced[left + 1] += column.val() * weight;
        }
        auto parentRow = out->writeLine(row.index());
        for (std::size_t node = 0; node < reduced.size(); ++node)
        {
            if (reduced[node] != typename VecDeriv::value_type{})
                parentRow.addCol(node, reduced[node]);
        }
    }
}

template<class DataTypes>
const sofa::linearalgebra::BaseMatrix* PiecewiseLinearStrainMapping<DataTypes>::getJ()
{
    return ensureReady() ? &m_jacobian : nullptr;
}

template<class DataTypes>
const sofa::linearalgebra::BaseMatrix* PiecewiseLinearStrainMapping<DataTypes>::getJ(
    const sofa::core::MechanicalParams*)
{
    return getJ();
}

template<class DataTypes>
const typename PiecewiseLinearStrainMapping<DataTypes>::Jacobians*
PiecewiseLinearStrainMapping<DataTypes>::getJs()
{
    return ensureReady() ? &m_jacobians : nullptr;
}

template class SOFA_COSSERAT_API PiecewiseLinearStrainMapping<sofa::defaulttype::Vec3Types>;
template class SOFA_COSSERAT_API PiecewiseLinearStrainMapping<sofa::defaulttype::Vec6Types>;

} // namespace Cosserat::mapping

namespace Cosserat
{
void registerPiecewiseLinearStrainMapping(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Continuous piecewise linear interpolation from nodal Cosserat strains to dependent samples")
        .add<mapping::PiecewiseLinearStrainMapping<sofa::defaulttype::Vec3Types>>(true)
        .add<mapping::PiecewiseLinearStrainMapping<sofa::defaulttype::Vec6Types>>());
}
} // namespace Cosserat
