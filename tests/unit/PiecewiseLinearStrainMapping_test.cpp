/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
* Copyright (c) 2026. Distributed under the GNU LGPL, version 2.1 or later.     *
* See <https://www.gnu.org/licenses/>.                                         *
******************************************************************************/
#include <Cosserat/mapping/PiecewiseLinearStrainMapping.h>
#include <Cosserat/forcefield/standard/BeamHookeLawForceField.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/testing/BaseSimulationTest.h>
#include <limits>
#include <cmath>
#include <algorithm>

namespace Cosserat::mapping
{

template<class DataTypes>
class PiecewiseLinearStrainMappingTest : public sofa::testing::BaseSimulationTest
{
public:
    using Mapping = PiecewiseLinearStrainMapping<DataTypes>;
    using State = sofa::component::statecontainer::MechanicalObject<DataTypes>;
    using Coord = typename DataTypes::Coord;
    using Vec = typename DataTypes::VecCoord;
    using Matrix = typename DataTypes::MatrixDeriv;
    using Abscissas = typename Mapping::Abscissas;
    static constexpr auto dimension = DataTypes::deriv_total_size;
    SceneInstance m_scene;
    typename Mapping::SPtr m_mapping;
    typename State::SPtr m_parent;
    typename State::SPtr m_child;

    void doSetUp() override
    {
        m_parent = sofa::core::objectmodel::New<State>();
        m_child = sofa::core::objectmodel::New<State>();
        m_scene.root->createChild("nodes")->addObject(m_parent);
        const auto childNode = m_scene.root->createChild("samples");
        childNode->addObject(m_child);
        m_mapping = sofa::core::objectmodel::New<Mapping>();
        childNode->addObject(m_mapping);
        m_parent->resize(3);
        m_parent->init();
        m_child->init();
        m_mapping->setModels(m_parent.get(), m_child.get());
        m_mapping->d_nodeAbscissa.setValue({0.0, 0.3, 1.0});
        m_mapping->d_sampleAbscissa.setValue({0.0, 0.15, 0.3, 0.65, 1.0});
        m_mapping->init();
    }

    Vec makeVector(const std::size_t count, const double offset) const
    {
        Vec values(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            for (std::size_t axis = 0; axis < dimension; ++axis)
            {
                values[i][axis] = offset + 0.23 * (i + 1) - 0.13 * axis;
            }
        }
        return values;
    }
};

using StrainTypes = ::testing::Types<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec6Types>;
TYPED_TEST_SUITE(PiecewiseLinearStrainMappingTest, StrainTypes);

TYPED_TEST(PiecewiseLinearStrainMappingTest, ReproducesAffineFieldOnNonuniformNodesAndEndpoints)
{
    using Vec = typename TestFixture::Vec;
    sofa::Data<Vec> input;
    sofa::Data<Vec> output;
    Vec values(3);
    const auto& nodes = this->m_mapping->d_nodeAbscissa.getValue();
    const auto& samples = this->m_mapping->d_sampleAbscissa.getValue();
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        for (std::size_t axis = 0; axis < TestFixture::dimension; ++axis)
        {
            values[i][axis] = 2.0 + axis + (axis + 1) * nodes[i];
        }
    }
    input.setValue(values);
    this->m_mapping->apply(nullptr, output, input);
    ASSERT_EQ(output.getValue().size(), samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i)
    {
        for (std::size_t axis = 0; axis < TestFixture::dimension; ++axis)
        {
            EXPECT_NEAR(output.getValue()[i][axis], 2.0 + axis + (axis + 1) * samples[i], 1e-13);
        }
    }
    EXPECT_TRUE(this->m_mapping->isLinear());
    EXPECT_EQ(this->m_parent->getSize(), 3u);
    EXPECT_EQ(this->m_child->getSize(), samples.size());
}

TYPED_TEST(PiecewiseLinearStrainMappingTest, VelocityMatchesFiniteDifferenceAndAssembledJacobian)
{
    using Vec = typename TestFixture::Vec;
    const Vec positions = this->makeVector(3, 0.4);
    const Vec velocity = this->makeVector(3, -0.7);
    constexpr double step = 1e-6;
    Vec perturbed = positions;
    for (std::size_t i = 0; i < positions.size(); ++i)
    {
        perturbed[i] += velocity[i] * step;
    }
    sofa::Data<Vec> x, xp, v, y, yp, mappedVelocity;
    x.setValue(positions);
    xp.setValue(perturbed);
    v.setValue(velocity);
    this->m_mapping->apply(nullptr, y, x);
    this->m_mapping->apply(nullptr, yp, xp);
    this->m_mapping->applyJ(nullptr, mappedVelocity, v);
    const auto* jacobian = this->m_mapping->getJ();
    ASSERT_NE(jacobian, nullptr);
    EXPECT_EQ(this->m_mapping->getJ(sofa::core::mechanicalparams::defaultInstance()), jacobian);
    ASSERT_EQ(this->m_mapping->getJs()->at(0), jacobian);
    for (std::size_t i = 0; i < y.getValue().size(); ++i)
    {
        for (std::size_t axis = 0; axis < TestFixture::dimension; ++axis)
        {
            const double expected = mappedVelocity.getValue()[i][axis];
            EXPECT_NEAR((yp.getValue()[i][axis] - y.getValue()[i][axis]) / step, expected, 1e-9);
            double product = 0;
            for (std::size_t j = 0; j < velocity.size(); ++j)
            {
                for (std::size_t c = 0; c < TestFixture::dimension; ++c)
                {
                    product += jacobian->element(i * TestFixture::dimension + axis,
                                                 j * TestFixture::dimension + c) * velocity[j][c];
                }
            }
            EXPECT_NEAR(product, expected, 1e-13);
        }
    }
}

TYPED_TEST(PiecewiseLinearStrainMappingTest, ForceTransposePreservesVirtualWorkAndAccumulates)
{
    using Vec = typename TestFixture::Vec;
    sofa::Data<Vec> velocity, mappedVelocity, force, parentForce;
    velocity.setValue(this->makeVector(3, 0.2));
    force.setValue(this->makeVector(5, 0.8));
    const Vec initial = this->makeVector(3, -0.4);
    parentForce.setValue(initial);
    this->m_mapping->applyJ(nullptr, mappedVelocity, velocity);
    this->m_mapping->applyJT(static_cast<const sofa::core::MechanicalParams*>(nullptr), parentForce, force);
    double childWork = 0;
    double parentWork = 0;
    for (std::size_t i = 0; i < force.getValue().size(); ++i)
    {
        childWork += force.getValue()[i] * mappedVelocity.getValue()[i];
    }
    for (std::size_t i = 0; i < initial.size(); ++i)
    {
        parentWork += (parentForce.getValue()[i] - initial[i]) * velocity.getValue()[i];
    }
    EXPECT_NEAR(childWork, parentWork, 1e-12);
    const Vec first = parentForce.getValue();
    this->m_mapping->applyJT(static_cast<const sofa::core::MechanicalParams*>(nullptr), parentForce, force);
    for (std::size_t i = 0; i < initial.size(); ++i)
    {
        EXPECT_LT((parentForce.getValue()[i] - (first[i] * 2 - initial[i])).norm(), 1e-12);
    }
}

TYPED_TEST(PiecewiseLinearStrainMappingTest, ConstraintTransposePreservesSparseRowsAndExistingEntries)
{
    using Vec = typename TestFixture::Vec;
    using Matrix = typename TestFixture::Matrix;
    const Vec forces = this->makeVector(5, 0.3);
    Matrix child;
    child.writeLine(7).addCol(1, forces[1]);
    child.writeLine(7).addCol(3, forces[3]);
    child.writeLine(19).addCol(4, forces[4]);
    Matrix parent;
    parent.writeLine(7).addCol(0, forces[0]);
    sofa::Data<Matrix> input, output;
    input.setValue(child);
    output.setValue(parent);
    this->m_mapping->applyJT(static_cast<const sofa::core::ConstraintParams*>(nullptr), output, input);
    const auto* jacobian = this->m_mapping->getJ();
    for (auto row = output.getValue().begin(); row != output.getValue().end(); ++row)
    {
        for (auto col = row.begin(); col != row.end(); ++col)
        {
            for (std::size_t axis = 0; axis < TestFixture::dimension; ++axis)
            {
                double expected = row.index() == 7 && col.index() == 0 ? forces[0][axis] : 0;
                const auto childRow = input.getValue().readLine(row.index());
                for (auto childCol = childRow.begin(); childCol != childRow.end(); ++childCol)
                {
                    expected += childCol.val()[axis] * jacobian->element(
                        childCol.index() * TestFixture::dimension + axis,
                        col.index() * TestFixture::dimension + axis);
                }
                EXPECT_NEAR(col.val()[axis], expected, 1e-13);
            }
        }
    }
}

TYPED_TEST(PiecewiseLinearStrainMappingTest, MapsRestStrainAndAcceleration)
{
    using Vec = typename TestFixture::Vec;
    const Vec values = this->makeVector(3, 0.6);
    this->m_parent->write(sofa::core::vec_id::write_access::restPosition)->setValue(values);
    this->m_mapping->init();
    sofa::Data<Vec> input, expected, acceleration, velocity;
    input.setValue(values);
    velocity.setValue(this->makeVector(3, 8.0));
    this->m_mapping->apply(nullptr, expected, input);
    this->m_mapping->computeAccFromMapping(nullptr, acceleration, velocity, input);
    const auto& rest = this->m_child->read(sofa::core::vec_id::read_access::restPosition)->getValue();
    for (std::size_t i = 0; i < expected.getValue().size(); ++i)
    {
        EXPECT_LT((rest[i] - expected.getValue()[i]).norm(), 1e-13);
        EXPECT_LT((acceleration.getValue()[i] - expected.getValue()[i]).norm(), 1e-13);
    }
}

TYPED_TEST(PiecewiseLinearStrainMappingTest, UpstreamMaterialHasExactPLSEnergyForceAndTangent)
{
    using Vec = typename TestFixture::Vec;
    using ForceField = sofa::component::forcefield::BeamHookeLawForceField<TypeParam>;
    this->m_parent->resize(2);
    this->m_mapping->d_nodeAbscissa.setValue({0.0, 1.0});
    const double gaussOffset = 1.0 / (2.0 * std::sqrt(3.0));
    this->m_mapping->d_sampleAbscissa.setValue({0.5 - gaussOffset, 0.5 + gaussOffset});
    this->m_mapping->init();
    ASSERT_EQ(this->m_child->read(sofa::core::vec_id::read_access::restPosition)->getValue().size(), 2u);
    const auto forceField = sofa::core::objectmodel::New<ForceField>();
    this->m_scene.root->getChild("samples")->addObject(forceField);
    ASSERT_TRUE(forceField->findData("length")->read("0.5 0.5"));
    ASSERT_TRUE(forceField->findData("youngModulus")->read("2000000"));
    ASSERT_TRUE(forceField->findData("poissonRatio")->read("0.3"));
    ASSERT_TRUE(forceField->findData("radius")->read("0.01"));
    forceField->init();
    sofa::core::MechanicalParams params;
    params.setKFactor(1.0);
    sofa::Data<Vec> input, samples, sampleForce, nodeForce, zeroVelocity;
    const Vec values = this->makeVector(2, 0.1);
    input.setValue(values);
    zeroVelocity.setValue(Vec(2));
    this->m_mapping->apply(&params, samples, input);
    const double pi = std::acos(-1.0);
    const double radius = 0.01;
    const double inertia = pi * std::pow(radius, 4) / 4;
    const double shear = 2000000 / 2.6;
    sofa::type::vector<double> stiffness{2 * shear * inertia, 2000000 * inertia, 2000000 * inertia};
    if constexpr (TestFixture::dimension == 6)
    {
        const double area = pi * radius * radius;
        stiffness.insert(stiffness.end(), {2000000 * area, shear * area, shear * area});
    }
    double expectedEnergy = 0;
    for (std::size_t axis = 0; axis < TestFixture::dimension; ++axis)
    {
        const double a = values[0][axis];
        const double b = values[1][axis];
        expectedEnergy += stiffness[axis] * (a * a + a * b + b * b) / 6;
    }
    EXPECT_NEAR(forceField->getPotentialEnergy(&params, samples), expectedEnergy,
                1e-11 * std::max(1.0, expectedEnergy));
    forceField->addForce(&params, sampleForce, samples, zeroVelocity);
    this->m_mapping->applyJT(&params, nodeForce, sampleForce);
    for (std::size_t node = 0; node < 2; ++node)
    {
        for (std::size_t axis = 0; axis < TestFixture::dimension; ++axis)
        {
            EXPECT_NEAR(nodeForce.getValue()[node][axis],
                -stiffness[axis] * (2 * values[node][axis] + values[1 - node][axis]) / 6,
                1e-11 * std::max(1.0, stiffness[axis]));
            Vec basis(2);
            basis[node][axis] = 1.0;
            sofa::Data<Vec> direction, sampleDirection, sampleDForce, nodeDForce;
            direction.setValue(basis);
            this->m_mapping->applyJ(&params, sampleDirection, direction);
            forceField->addDForce(&params, sampleDForce, sampleDirection);
            this->m_mapping->applyJT(&params, nodeDForce, sampleDForce);
            EXPECT_NEAR(nodeDForce.getValue()[node][axis], -stiffness[axis] / 3,
                        1e-11 * std::max(1.0, stiffness[axis]));
            EXPECT_NEAR(nodeDForce.getValue()[1 - node][axis], -stiffness[axis] / 6,
                        1e-11 * std::max(1.0, stiffness[axis]));
        }
    }
}

TYPED_TEST(PiecewiseLinearStrainMappingTest, DataEditsRebuildWeightsAndJacobian)
{
    this->m_mapping->d_sampleAbscissa.setValue({1.0, 0.0, 0.3});
    const auto* jacobian = this->m_mapping->getJ();
    ASSERT_NE(jacobian, nullptr);
    EXPECT_EQ(this->m_child->getSize(), 3u);
    EXPECT_DOUBLE_EQ(jacobian->element(0, 2 * TestFixture::dimension), 1.0);
    this->m_mapping->d_nodeAbscissa.setValue({0.0, 0.6, 1.0});
    jacobian = this->m_mapping->getJ();
    ASSERT_NE(jacobian, nullptr);
    EXPECT_DOUBLE_EQ(jacobian->element(2 * TestFixture::dimension, 0), 0.5);
}

TYPED_TEST(PiecewiseLinearStrainMappingTest, RejectsInvalidCoordinatesAndRecovers)
{
    using Abscissas = typename TestFixture::Abscissas;
    const sofa::type::vector<Abscissas> badNodes = {
        {}, {0.0}, {0.0, 0.0, 1.0}, {0.0, 1.0, 0.3},
        {0.0, std::numeric_limits<double>::quiet_NaN(), 1.0},
        {0.0, 0.3, std::numeric_limits<double>::infinity()}};
    for (const auto& nodes : badNodes)
    {
        EXPECT_MSG_EMIT(Error);
        this->m_mapping->d_nodeAbscissa.setValue(nodes);
        EXPECT_EQ(this->m_mapping->getJ(), nullptr);
    }
    this->m_mapping->d_nodeAbscissa.setValue({0.0, 0.3, 1.0});
    ASSERT_NE(this->m_mapping->getJ(), nullptr);
    for (const double sample : {-0.1, 1.1, std::numeric_limits<double>::quiet_NaN()})
    {
        EXPECT_MSG_EMIT(Error);
        this->m_mapping->d_sampleAbscissa.setValue({sample});
        EXPECT_EQ(this->m_mapping->getJ(), nullptr);
    }
    this->m_mapping->d_sampleAbscissa.setValue({0.0, 1.0});
    EXPECT_NE(this->m_mapping->getJ(), nullptr);
}

TYPED_TEST(PiecewiseLinearStrainMappingTest, EmptySamplesAndResizedInputAreHandled)
{
    this->m_mapping->d_sampleAbscissa.setValue({});
    ASSERT_NE(this->m_mapping->getJ(), nullptr);
    EXPECT_EQ(this->m_child->getSize(), 0u);
    {
        EXPECT_MSG_EMIT(Error);
        this->m_parent->resize(4);
        EXPECT_EQ(this->m_mapping->getJ(), nullptr);
    }
    this->m_mapping->d_nodeAbscissa.setValue({0.0, 0.3, 0.6, 1.0});
    EXPECT_NE(this->m_mapping->getJ(), nullptr);
}

} // namespace Cosserat::mapping
