// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FSI_DIRICHLETNEUMANN_FACTORY_HPP
#define FOUR_C_FSI_DIRICHLETNEUMANN_FACTORY_HPP

#include "4C_config.hpp"

#include "4C_utils_parameter_list.fwd.hpp"

#include <mpi.h>



FOUR_C_NAMESPACE_OPEN

namespace FSI
{
  class DirichletNeumann;
  /**
   *  \brief Factory that creates the appropriate DirichletNeumann algorithm
   *
   *  To create a DirichletNeumann algorithm, call the static create_algorithm function directly! No
   * instance of DirichletNeumannFactory has to be created! If you try to call the constructor, you
   * will get an error message, since it is set to be private.
   */
  class DirichletNeumannFactory
  {
   private:
    /// constructor
    DirichletNeumannFactory() {};

   public:
    /**
     *  \brief Creates the appropriate DirichletNeumann algorithm
     *
     * This function is static so that it can be called without creating a factory object first.
     * It can be called directly.
     *
     * \param[in] comm MPI Communicator used in FSI::Partitioned for Terminal Output
     * \param[in] fsidyn List of FSI Input parameters
     *
     * \return Coupling algorithm based on Dirichlet-Neumann partitioning
     */
    static std::shared_ptr<DirichletNeumann> create_algorithm(
        MPI_Comm comm, const Teuchos::ParameterList& fsidyn);
  };
}  // namespace FSI

FOUR_C_NAMESPACE_CLOSE

#endif
