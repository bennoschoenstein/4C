// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_DISCRETIZATION_HDG_HPP
#define FOUR_C_FEM_DISCRETIZATION_HDG_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization_faces.hpp"
#include "4C_fem_discretization_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <memory>
#include <string>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class MapExtractor;
  class SparseMatrix;
}  // namespace Core::LinAlg

namespace Core::FE
{
  enum class HDGAction
  {
    project_dirich_field  ///< project dirichlet field
  };

  class DiscretizationHDG : public DiscretizationFaces
  {
   public:
    /*!
    \brief Standard Constructor

    \param name: name of this discretization
    \param comm : MPI comm object associated with this discretization
    \param n_dim: number of space dimensions of this discretization
    */
    DiscretizationHDG(const std::string name, MPI_Comm comm, unsigned int n_dim);


    /*!
    \brief Complete construction of a discretization  (Filled()==true NOT prerequisite)

    After adding or deleting nodes or elements or redistributing them in parallel,
    or adding/deleting boundary conditions, this method has to be called to (re)construct
    pointer topologies.<br>
    It builds in this order:<br>
    Standard fill_complete of base class
    - row map of nodes
    - column map of nodes
    - row map of elements
    - column map of elements
    - pointers from elements to nodes
    - pointers from nodes to elements
    - assigns degrees of freedoms
    - map of element register classes
    - calls all element register initialize methods
    - build geometries of all Dirichlet and Neumann boundary conditions

    Additional features
    - build internal faces elements
    - build maps and pointers for internal faces

    \param assigndegreesoffreedom (in) : if true, resets existing dofsets and performs
                                         assigning of degrees of freedoms to nodes and
                                         elements.
    \param initelements (in) : if true, build element register classes and call initialize()
                               on each type of finite element present
    \param doboundaryconditions (in) : if true, build geometry of boundary conditions
                                       present.

    \note In order to receive a fully functional discretization, this method must be called
          with all parameters set to true (the default). The parameters though can be
          used to turn off specific tasks to allow for more flexibility in the
          construction of a discretization, where it is known that this method will
          be called more than once.

    \note Sets Filled()=true
    */
    int fill_complete(OptionsFillComplete options = {}) override;

    /*!
     * this function has the same functionality as the function in the base class,
     * additionally, the degree of the elements is communicated, such that ghosted elements
     * also have full knowledge about the face degrees. This is necessary for discretizations
     * with nonuniform degree distributions and p-adaptivity
     *
     *  schoeder 06/14
     */
    void assign_global_ids(MPI_Comm comm,
        const std::map<std::vector<int>, std::shared_ptr<Core::Elements::Element>>& elementmap,
        std::map<int, std::shared_ptr<Core::Elements::Element>>& finalelements) override;

  };  // class DiscretizationHDG

  /** \brief Specialized Dbc evaluation class for HDG discretizations
   *
   *  */
  class DbcHDG : public Dbc
  {
   public:
    /// constructor
    DbcHDG() {};

   protected:
    void read_dirichlet_condition(const Teuchos::ParameterList& params,
        const Core::FE::Discretization& discret, const Core::Conditions::Condition& cond,
        double time, DbcInfo& info, const std::shared_ptr<std::set<int>>* dbcgids,
        int hierarchical_order) const override;

    void read_dirichlet_condition(const Teuchos::ParameterList& params,
        const Core::FE::DiscretizationFaces& discret, const Core::Conditions::Condition& cond,
        double time, DbcInfo& info, const std::shared_ptr<std::set<int>>* dbcgids,
        int hierarchical_order) const;

    void do_dirichlet_condition(const Teuchos::ParameterList& params,
        const Core::FE::Discretization& discret, const Core::Conditions::Condition& cond,
        double time, const std::shared_ptr<Core::LinAlg::Vector<double>>* systemvectors,
        const Core::LinAlg::Vector<int>& toggle,
        const std::shared_ptr<std::set<int>>* dbcgids) const override;

    void do_dirichlet_condition(const Teuchos::ParameterList& params,
        const Core::FE::DiscretizationFaces& discret, const Core::Conditions::Condition& cond,
        double time, const std::shared_ptr<Core::LinAlg::Vector<double>>* systemvectors,
        const Core::LinAlg::Vector<int>& toggle) const;
  };  // class DbcHDG
}  // namespace Core::FE

/// << operator
std::ostream& operator<<(std::ostream& os, const Core::FE::DiscretizationHDG& dis);


FOUR_C_NAMESPACE_CLOSE

#endif
