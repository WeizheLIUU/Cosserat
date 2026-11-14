/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
* Copyright (c) 2026. Distributed under the GNU LGPL, version 2.1 or later.     *
* See <https://www.gnu.org/licenses/>.                                         *
******************************************************************************/
#pragma once

#include <liegroups/SE3.h>
#include <Cosserat/mapping/SofaLieGroupsUtils.h>
#include <vector>
#include <algorithm>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/visual/DrawTool.h>
#include <sofa/helper/logging/Messaging.h>
#include <cmath>
#include <limits>

namespace Cosserat::mapping::pls_detail
{
using Matrix6 = Eigen::Matrix<double, 6, 6>;
using Vector6 = Eigen::Matrix<double, 6, 1>;
using SE3 = sofa::component::cosserat::liegroups::SE3<double>;

inline Eigen::Matrix3d skew(const Eigen::Vector3d& vector)
{
    Eigen::Matrix3d matrix;
    matrix << 0, -vector.z(), vector.y(), vector.z(), 0, -vector.x(),
              -vector.y(), vector.x(), 0;
    return matrix;
}

inline SE3 exponential(Vector6 strain, double length)
{
    strain(3) += 1.0;
    return SE3::computeExp(strain * length);
}

inline Matrix6 tangent(const Vector6& strain, double length)
{
    if (!std::isfinite(length) || !strain.allFinite())
    {
        msg_error("PLSKinematics") << "Cannot evaluate a non-finite strain or length.";
        return Matrix6::Zero();
    }
    Matrix6 A = Matrix6::Zero();
    A.topLeftCorner<3, 3>() = skew(strain.head<3>());
    A.bottomRightCorner<3, 3>() = A.topLeftCorner<3, 3>();
    A.bottomLeftCorner<3, 3>() = skew(Eigen::Vector3d(1 + strain(3), strain(4), strain(5)));
    A *= length;
    double norm = A.cwiseAbs().rowwise().sum().maxCoeff();
    if (!std::isfinite(norm))
    {
        msg_error("PLSKinematics") << "The tangent generator overflowed.";
        return Matrix6::Zero();
    }
    int squarings = 0;
    while (norm > 0.5)
    {
        A *= 0.5;
        norm *= 0.5;
        ++squarings;
    }
    const Matrix6 identity = Matrix6::Identity();
    Matrix6 exponential = identity;
    Matrix6 phi = identity;
    Matrix6 term = identity;
    for (int n = 1; n <= 64; ++n)
    {
        term = (term * A / static_cast<double>(n)).eval();
        exponential += term;
        phi += term / static_cast<double>(n + 1);
        if (term.cwiseAbs().maxCoeff() <= std::numeric_limits<double>::epsilon()
            * exponential.cwiseAbs().maxCoeff())
        {
            break;
        }
    }
    for (int i = 0; i < squarings; ++i)
    {
        phi = (0.5 * (identity + exponential) * phi).eval();
        exponential = (exponential * exponential).eval();
    }
    return length * phi;
}

template<class Sections, class Frames>
void updateTangents(Sections& sections, Frames& frames)
{
    if (sections.empty())
    {
        return;
    }
    sections[0].setTanAdjointMatrix(Matrix6::Zero());
    for (std::size_t i = 1; i < sections.size(); ++i)
    {
        auto& section = sections[i];
        section.setTanAdjointMatrix(tangent(section.getStrainsVec(), section.getLength()));
    }
    for (auto& frame : frames)
    {
        const auto index = frame.getRelatedSectionIndex();
        if (static_cast<std::size_t>(index) < sections.size())
        {
            frame.setTanAdjointMatrix(tangent(sections[index].getStrainsVec(),
                                              frame.getDistanceToSectionStart()));
        }
    }
}

/// Directional time derivatives along q(t)=q+t*v. Derivatives, not Taylor
/// coefficients: second stores d²/dt². No finite-difference step is involved.
struct TransformJet
{
    Eigen::Matrix4d value = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d first = Eigen::Matrix4d::Zero();
    Eigen::Matrix4d second = Eigen::Matrix4d::Zero();
};

inline TransformJet compose(const TransformJet& a, const TransformJet& b)
{
    return {a.value * b.value,
            a.first * b.value + a.value * b.first,
            a.second * b.value + 2.0 * a.first * b.first + a.value * b.second};
}

inline TransformJet exponentialJet(Vector6 strain, const Vector6& velocity, double length)
{
    strain[3] += 1.0;
    Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
    Eigen::Matrix4d V = Eigen::Matrix4d::Zero();
    A.topLeftCorner<3, 3>() = length * skew(strain.head<3>());
    A.topRightCorner<3, 1>() = length * strain.tail<3>();
    V.topLeftCorner<3, 3>() = length * skew(velocity.head<3>());
    V.topRightCorner<3, 1>() = length * velocity.tail<3>();
    if (!A.allFinite() || !V.allFinite())
    {
        TransformJet invalid;
        invalid.second.setConstant(std::numeric_limits<double>::quiet_NaN());
        return invalid;
    }
    double norm = A.cwiseAbs().rowwise().sum().maxCoeff();
    if (!std::isfinite(norm))
    {
        TransformJet invalid;
        invalid.second.setConstant(std::numeric_limits<double>::quiet_NaN());
        return invalid;
    }
    int squarings = 0;
    while (norm > 0.5)
    {
        A *= 0.5;
        V *= 0.5;
        norm *= 0.5;
        ++squarings;
    }
    TransformJet result;
    TransformJet term;
    constexpr double epsilon = std::numeric_limits<double>::epsilon();
    for (int n = 1; n <= 64; ++n)
    {
        const double divisor = static_cast<double>(n);
        const Eigen::Matrix4d second = (term.second * A + 2.0 * term.first * V) / divisor;
        const Eigen::Matrix4d first = (term.first * A + term.value * V) / divisor;
        term.value = (term.value * A / divisor).eval();
        term.first = first;
        term.second = second;
        result.value += term.value;
        result.first += term.first;
        result.second += term.second;
        if (term.value.cwiseAbs().maxCoeff() <= epsilon * result.value.cwiseAbs().maxCoeff()
            && term.first.cwiseAbs().maxCoeff() <= epsilon * result.first.cwiseAbs().maxCoeff()
            && term.second.cwiseAbs().maxCoeff() <= epsilon * result.second.cwiseAbs().maxCoeff())
            break;
    }
    for (int i = 0; i < squarings; ++i)
        result = compose(result, result);
    return result;
}

template<class Strains, class Velocities, class BaseCoord, class BaseVelocity,
         class Sections, class Frames>
bool frameMotionJets(const Strains& q, const Velocities& v,
                     const BaseCoord& base, const BaseVelocity& baseVelocity,
                     const Sections& sections, const Frames& frames,
                     std::vector<TransformJet>& output)
{
    if (q.size() != v.size() || sections.size() != q.size() + 1)
        return false;
    TransformJet baseJet;
    const auto R = sofaQuatToEigen(base.getOrientation()).toRotationMatrix();
    const Eigen::Vector3d omega(baseVelocity[3], baseVelocity[4], baseVelocity[5]);
    const Eigen::Matrix3d W = skew(omega);
    baseJet.value.template topLeftCorner<3, 3>() = R;
    baseJet.first.template topLeftCorner<3, 3>() = W * R;
    baseJet.second.template topLeftCorner<3, 3>() = W * W * R;
    for (int c = 0; c < 3; ++c)
    {
        baseJet.value(c, 3) = base.getCenter()[c];
        baseJet.first(c, 3) = baseVelocity[c];
    }
    std::vector<Vector6> strains(q.size()), rates(q.size());
    std::vector<TransformJet> prefix(q.size() + 1);
    prefix[0] = baseJet;
    for (std::size_t k = 0; k < q.size(); ++k)
    {
        strains[k].setZero();
        rates[k].setZero();
        for (unsigned int c = 0; c < q[k].size(); ++c)
        {
            strains[k][c] = q[k][c];
            rates[k][c] = v[k][c];
        }
        if (!strains[k].allFinite() || !rates[k].allFinite())
            return false;
        prefix[k + 1] = compose(prefix[k], exponentialJet(strains[k], rates[k],
                                                         sections[k + 1].getLength()));
    }
    output.resize(frames.size());
    for (std::size_t f = 0; f < frames.size(); ++f)
    {
        const auto section = static_cast<std::size_t>(frames[f].getRelatedSectionIndex());
        if (section == 0)
            output[f] = baseJet;
        else if (section <= q.size())
            output[f] = compose(prefix[section - 1], exponentialJet(strains[section - 1],
                rates[section - 1], frames[f].getDistanceToSectionStart()));
        else
            return false;
        if (!output[f].value.allFinite() || !output[f].first.allFinite()
            || !output[f].second.allFinite())
            return false;
    }
    return true;
}

/// World-wrench to strain/base blocks, reused for every sparse projection row.
/// These are the same adjoint/tangent products as the native reverse sweep.
/// They are not a frozen Jacobian: the owning mapping invalidates them on each
/// geometry update and every output-position edit (including base motion).
template<int N>
struct ConstraintTransposeCache
{
    using Block = Eigen::Matrix<double, N, 6>;
    std::vector<std::vector<Block>> strain;
    std::vector<Matrix6> base;

    template<class Sections, class Frames, class Positions, class Indices>
    void update(const Sections& sections, const Frames& frames,
                const Positions& positions, const Indices& indices)
    {
        strain.resize(frames.size());
        base.resize(frames.size());
        if (positions.empty())
            return;
        const auto basePose = rigidCoordToSE3(positions[0]);
        const Matrix6 baseProjector = basePose.buildProjectionMatrix(basePose.rotation().matrix());
        for (std::size_t f = 0; f < frames.size(); ++f)
        {
            const auto sectionIndex = static_cast<std::size_t>(indices[f]);
            auto& blocks = strain[f];
            blocks.resize(sectionIndex);
            const auto pose = rigidCoordToSE3(positions[f]);
            const Matrix6 projector = pose.buildProjectionMatrix(pose.rotation().matrix());
            Matrix6 wrench = frames[f].getCoAdjoint() * projector.transpose();
            if (sectionIndex > 0)
            {
                blocks[sectionIndex - 1] =
                    frames[f].getTangAdjointMatrix().transpose().template topRows<N>() * wrench;
                for (std::size_t k = sectionIndex - 1; k > 0; --k)
                {
                    wrench = (sections[k].getCoAdjoint() * wrench).eval();
                    blocks[k - 1] =
                        sections[k].getTangAdjointMatrix().transpose().template topRows<N>() * wrench;
                }
                wrench = (sections[0].getCoAdjoint() * wrench).eval();
            }
            base[f] = baseProjector * wrench;
        }
    }

    template<class InputMatrix, class StrainMatrix, class BaseMatrix>
    void apply(const InputMatrix& in, StrainMatrix& outStrain,
               BaseMatrix& outBase, unsigned int baseIndex) const
    {
        for (auto row = in.begin(); row != in.end(); ++row)
        {
            auto strainRow = outStrain.writeLine(row.index());
            auto baseRow = outBase.writeLine(row.index());
            for (auto col = row.begin(); col != row.end(); ++col)
            {
                const auto frame = static_cast<std::size_t>(col.index());
                Vector6 force;
                for (int c = 0; c < 6; ++c)
                    force[c] = col.val()[c];
                for (std::size_t k = 0; k < strain[frame].size(); ++k)
                {
                    const Eigen::Matrix<double, N, 1> value = strain[frame][k] * force;
                    if (!value.isZero(0))
                    {
                        typename StrainMatrix::Data derivative;
                        for (int c = 0; c < N; ++c)
                            derivative[c] = value[c];
                        strainRow.addCol(k, derivative);
                    }
                }
                const Vector6 value = base[frame] * force;
                typename BaseMatrix::Data derivative;
                for (int c = 0; c < 6; ++c)
                    derivative[c] = value[c];
                baseRow.addCol(baseIndex, derivative);
            }
        }
    }
};

template<class Frames, class Strains, class Indices, class Color, class Evaluator>
void drawFrames(const sofa::core::visual::VisualParams* params, const Frames& frames,
                const Strains& strains, const Indices& indices, double radius,
                const Color& baseColor, int axis, bool uniformColor, Evaluator evaluator)
{
    if (!params->displayFlags().getShowMechanicalMappings() || frames.size() < 2)
    {
        return;
    }
    const auto state = params->drawTool()->makeStateLifeCycle();
    for (std::size_t i = 0; i + 1 < frames.size(); ++i)
    {
        const auto& from = frames[i].getCenter();
        const auto& to = frames[i + 1].getCenter();
        params->drawTool()->drawCylinder(from, to, radius, baseColor);
        auto color = baseColor;
        if (!uniformColor && i < indices.size())
        {
            const auto index = indices[i];
            if (index > 0 && static_cast<std::size_t>(index) <= strains.size()
                && axis >= 0 && static_cast<std::size_t>(axis) < strains[index - 1].size())
            {
                color = evaluator(strains[index - 1][axis]);
            }
        }
        params->drawTool()->drawLine(from, to, color);
    }
}
} // namespace Cosserat::mapping::pls_detail
