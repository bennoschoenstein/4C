// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_CONDITION_LOCSYS_HPP
#define FOUR_C_FEM_CONDITION_LOCSYS_HPP

#include "4C_config.hpp"

#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_utils_function_manager.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class SparseMatrix;
  class MultiMapExtractor;
  class BlockSparseMatrixBase;
}  // namespace Core::LinAlg

namespace Core::Conditions
{
  /*!
  \brief Class controlling local coordinate systems on points, lines and surfaces
         and supplying all necessary transformation methods for parallel vectors
         and matrices.


       The most general approach to apply Dirichlet conditions in a rotated, local system would
  be:

       1) Transform the system into local coordinates by means of
          \f[
            K \cdot D = F \rightarrow \tilde{K} \cdot \tilde{D} = \tilde{F}
          \f]
          with
          \f[
            \tilde{K} = Q \cdot K \cdot Q^T, \quad \tilde{F} = Q \cdot F, \quad \tilde{D} = Q
  \cdot D \f]

       2) Apply Dirichlet conditions in the rotated system in the standard way (Dirichlet Line: 1
  at diagonal, zeros else)

       3) Transform the system back into global coordinates, i.e.
          \f[
            \tilde{K} \cdot \tilde{D} = \tilde{F} \rightarrow K \cdot D = F
          \f]
          with
          \f[
            K = Q^T \cdot \tilde{K} \cdot Q, \quad F = Q^T \cdot \tilde{F}, \quad D = Q^T \cdot
  \tilde{D} \f]

          Here the transformation matrix #trafo_ is denoted \f$Q\f$.

       Nevertheless, we apply a more efficient algorithm which can be shown,to deliver an
  equivalent system of equations:

       1) Therefore we only apply one left transformation to our system of equations according
          \f[
            K \cdot D = F \rightarrow Q \cdot K \cdot D = Q \cdot F
          \f]

       2) Afterwards we apply the rotated Dirichlet conditions in an appropriate manner

          We don't invert the left transformation of our system afterwards. This means, that we
  don't solve the original but an algebraic manipulated system of equations. Nevertheless we still
  solve for the original, non-rotated DoFs D.

  */
  class LocsysManager
  {
   public:
    //! @name Enums and Friends

    //@}


    /*!
     * \brief Standard Constructor
     *
     * \param discret (in): A discretization containing locsys boundary conditions
     * \param transformleftonly (in): Only a transformation from the left is going
     *                                to be applied on sysmatrix, if true
     *
     */
    explicit LocsysManager(Core::FE::Discretization& discret, int dim);

    /*!
     * Set current @p time and @p nodenormals to the locsys manager. The vector of @p
     * nodenormals my remain empty. It is only required for
     * calc_rotation_vector_for_normal_system().
     */
    void update(double time, std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> nodenormals,
        const Core::Utils::FunctionManager& function_manager);

    /*!
     *\brief Print this Manager
     *
     */
    void print() const;

    /*!
     * \brief Get MPI communicator
     *
     */
    inline MPI_Comm get_comm() const;

    //! @name Access methods

    /*!
     * \brief Get discretization
     *
     */
    inline Core::FE::Discretization& discret() const { return discret_; };

    /*!
     * \brief Get problem dimension
     *
     */
    inline const int& n_dim() { return dim_; };

    /*!
     * \brief Get number of local system conditions
     *
     */
    inline int num_locsys() const { return numlocsys_; };

    /*!
     * \brief Get types of local system conditions
     *
     */
    inline std::vector<Core::Conditions::ConditionType> type_locsys() const { return typelocsys_; };

    /*!
     * \brief Get type of a specific local system condition
     *
     */
    inline Core::Conditions::ConditionType type_locsys(int k) const
    {
      if (k >= numlocsys_) FOUR_C_THROW("Invalid vector index");
      return typelocsys_[k];
    };

    /*!
     * \brief Retrieve the global transformation matrix
     *
     */
    std::shared_ptr<const Core::LinAlg::SparseMatrix> trafo() const { return trafo_; }

    //@}

    //! @name Evaluation methods

    /*!
     * \brief Apply forward transformation of linear system of equations.
     *
     * This method transform the matrix #sysmat from global co-ordinate
     * into local co-ordinate systems, i.e.
     *   \f[ \tilde{K} = Q \cdot K \cdot Q^T \f]
     * in which \f$K\f$ is the globally  and \f$\tilde{K}\f$ the locally oriented matrix,
     * respectively. The transformation matrix #trafo_ is denoted \f$Q : D \mapsto \tilde{D}\f$.
     * The similar thing is done for the right-hand-side vector:
     *   \f[ \tilde{R} = Q \cdot R \f]
     */
    void rotate_global_to_local(
        std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat,  ///< systemmatrix, will be transformed
        Core::LinAlg::Vector<double>& rhs  ///< right-hand-side vector, will be transformed
    ) const;

    /*!
     * \brief Apply forward transformation of a single matrix
     *
     */
    void rotate_global_to_local(std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat) const;

    /*!
     * \brief Apply forward transformation of a single vector
     *
     */
    void rotate_global_to_local(Core::LinAlg::Vector<double>& vec, bool offset = false) const;

    /*!
     * \brief Apply backward transformation of result and linear system of equations
     *
     */
    void rotate_local_to_global(Core::LinAlg::Vector<double>& result,
        std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat,
        Core::LinAlg::Vector<double>& rhs) const;

    /*!
     * \brief Apply backward transformation of a single vector
     *
     */
    void rotate_local_to_global(Core::LinAlg::Vector<double>& vec, bool offset = false) const;

    /*!
     * \brief Apply backward transformation of a matrix
     *
     */
    void rotate_local_to_global(Core::LinAlg::SparseMatrix& sysmat) const;

    /*!
     * \brief Calculate rotation vector for (mass-consistent) normal system
     *
     */
    void calc_rotation_vector_for_normal_system(int numLocsysCond, double time);

    //@}

   private:
    // don't want = operator and cctor
    LocsysManager operator=(const LocsysManager& old);
    LocsysManager(const LocsysManager& old);

    /// @name Private Attributes
    //@{

    /// current discretization
    Core::FE::Discretization& discret_;

    /// id of locsys condition
    std::vector<int> id_;

    /// problem dimension
    int dim_;

    /// local system conditions
    std::vector<const Core::Conditions::Condition*> locsysconds_;

    /// number of local systems
    int numlocsys_;

    /// list of Node Normals for massConsistent BC
    std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> nodenormals_;

    /// types of local system conditions
    std::vector<Core::Conditions::ConditionType> typelocsys_;

    /// vector that indicates the existence of time curves or functions within the locsys
    /// condition
    bool locsysfunct_;

    /// maps the GID of a node onto the pseudo-rotation vector which rotates the global xyz system
    /// onto local system of the node
    std::map<int, Core::LinAlg::Matrix<3, 1>> nodalrotvectors_;

    /// assignment of local systems to nodes
    std::shared_ptr<Core::LinAlg::Vector<double>> locsystoggle_;

    /// maps containing the DOFs affected by locsys
    std::shared_ptr<Core::LinAlg::Map> locsysdofmap_;

    /// Transformation matrix which maps globally oriented components
    /// into locally oriented components (dubbed 'forward' transformation)
    ///
    /// Even if this matrix bears the general name transformation matrix here,
    /// one should be aware of that it is treated like a rotational
    /// transformation matrix. This means: The inverse (or 'backward') mapping
    /// is simply implemented as its transpose rather than a general inverse.
    std::shared_ptr<Core::LinAlg::SparseMatrix> trafo_;

    /// Transformation 'sub'-matrix with non-identity entries
    ///
    /// This is actually not a sub-matrix, but a global matrix
    /// with nil entries at (a lot) places
    std::shared_ptr<Core::LinAlg::SparseMatrix> subtrafo_;

    // Boolean, which is yes if a locsys warning has already been thrown
    bool warning_thrown_;
    //@}

  };  // class LocsysManager
}  // namespace Core::Conditions


FOUR_C_NAMESPACE_CLOSE

#endif
