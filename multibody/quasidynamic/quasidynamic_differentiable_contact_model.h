#pragma once

#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace multibody {
namespace quasidynamic {

/**
 @p QuasidynamicDifferentiableContactModel describes the dynamics of multibody
 systems as quasidynamic. The model steps at a fixed time step. The contact
 dynamics can be smoothed for differentiability.
 @tparam_nonsymbolic_scalar
 */
template <typename T>
class QuasidynamicDifferentiableContactModel : public systems::LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QuasidynamicDifferentiableContactModel);

  /** Returns the number of contact pairs. */
  int num_contacts() const { return num_contacts_; }

  /** Returns the dimension of the contact, may be 1, 2, or 3. */
  int contact_dim() const { return contact_dim_; }

  /** Gets the dynamics smoothing parameter. */
  double get_dynamics_smoothing(const systems::Context<T>& context) const;

  /** Sets the dynamics smoothing parameter. (Zero means no smoothing and larger
   values mean higher smoothing.)
   @pre `dynamics_smoothing >= 0`. */
  void set_dynamics_smoothing(systems::Context<T>* context,
                              double dynamics_smoothing) const;

  /** Returns the actuation input port */
  const systems::InputPort<T>& get_actuation_input_port() const;

  /** Returns the state output port. */
  const systems::OutputPort<T>& get_state_output_port() const;

  /** Returns the contact force output port. */
  const systems::OutputPort<T>& get_contact_force_output_port() const;

  /** Returns a reference to the vector of state in a given context.
   @pre `context` is compatible with this system. */
  const VectorX<T>& GetState(const systems::Context<T>& context) const;

  /** Sets the state in a given Context from a given vector.
   @pre `context` is compatible with this system.
   @pre `x` has the correct size. */
  void SetState(systems::Context<T>* context,
                const Eigen::Ref<const VectorX<T>>& x) const;

  /** Returns a reference to the disturbance matrix in a given context.
   @pre `context` is compatible with this system. */
  const MatrixX<T>& GetDisturbanceMatrix(
      const systems::Context<T>& context) const;

  /** Evaluates the P(x) matrix governing the quasidynamic model. */
  virtual const MatrixX<T>& EvalQuasidynamicPMatrix(
      const systems::Context<T>& context) const = 0;

  /** Evaluates the b(x,u) vector governing the quasidynamic model. */
  virtual const VectorX<T>& EvalQuasidynamicBVector(
      const systems::Context<T>& context) const = 0;

  /** Evaluates the contact jacobian and signed distance. */
  virtual const MatrixX<T>& EvalContactJacobian(
      const systems::Context<T>& context) const = 0;

  /** Evaluates the contact signed distance. */
  virtual const VectorX<T>& EvalContactSignedDistance(
      const systems::Context<T>& context) const = 0;

  /** Evaluates the friction coefficient for each contact. */
  virtual const VectorX<double>& EvalContactFrictionCoefficient(
      const systems::Context<T>& context) const = 0;

 protected:
  /** Constructur invoked by derived classes.
   @param converter Set to systems::SystemTypeTag<DerivedClass>{}.
   @param kappa The dynamics smoothing parameter.
   @pre `dynamics_smoothing >= 0`. */
  QuasidynamicDifferentiableContactModel(
      systems::SystemScalarConverter converter,
      double dynamics_smoothing = 0.0);

  /** Declares the states and input output ports.
   @param x The default state.
   @param num_inputs Number of inputs.
   @param num_contacts Number of contacts.
   @param contact_dim Dimension of contact, can be 1, 2, or 3.
   @param dt Time step size.

   @pre `x.size() > 0`.
   @pre `num_inputs > 0`.
   @pre `num_contacts > 0`.
   @pre `contact_dim == 1 || contact_dim == 2 || contact_dim == 3`.
   @pre `dt > 0`.*/
  void DeclareStateAndPorts(const Eigen::Ref<const VectorX<T>>& x,
                            int num_inputs, int num_contacts, int contact_dim,
                            double dt);

 private:
  void UpdateState(const systems::Context<T>& context,
                   systems::DiscreteValues<T>* values) const;

  void CalcLambdaOutput(const systems::Context<T>& context,
                        systems::BasicVector<T>* output) const;

  int num_contacts_{};
  int contact_dim_{};
  systems::NumericParameterIndex dynamics_smoothing_param_index_{};
  systems::DiscreteStateIndex x_state_index_{};
  systems::OutputPortIndex x_output_index_{};
  systems::InputPortIndex u_input_index_{};
  systems::OutputPortIndex lambda_output_index_{};
  systems::CacheIndex lambda_cache_index_{};
  systems::CacheIndex E_cache_index_{};
};

}  // namespace quasidynamic
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::quasidynamic::
        QuasidynamicDifferentiableContactModel);
