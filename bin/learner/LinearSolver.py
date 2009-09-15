##############################################################################
# This file is part of pysgpp, a program package making use of spatially    #
# adaptive sparse grids to solve numerical problems                         #
#                                                                           #
# Copyright (C) 2009 Valeriy Khakhutskyy (khakhutv@in.tum.de)               #
#                                                                           #
# pysgpp is free software; you can redistribute it and/or modify            #
# it under the terms of the GNU General Public License as published by      #
# the Free Software Foundation; either version 3 of the License, or         #
# (at your option) any later version.                                       #
#                                                                           #
# pysgpp is distributed in the hope that it will be useful,                 #
# but WITHOUT ANY WARRANTY; without even the implied warranty of            #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             #
# GNU Lesser General Public License for more details.                       #
#                                                                           #
# You should have received a copy of the GNU General Public License         #
# along with pysgpp; if not, write to the Free Software                     #
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA #
# or see <http://www.gnu.org/licenses/>.                                    #
#############################################################################

## @package LinearSolver
# @ingroup bin.learner
# @brief Abstract class for solving of linear equations
# @version $HEAD$ 


class LinearSolver(object):
    eventControllers = None   #list of object listening to the solver events
    
    def __init__(self):
        self.eventControllers = []

    
    ## Solver linear system
    # this method is not implemented
    #
    # @param linearSystem: DMSystemMatrix object of Linear System to solve
    def solve(self, linearSystem):
        raise NotImplementedError

    
    ## Add observer to the list
    #
    # @param observer: ProgressInfoPresentor object
    def attachEventController(self, observer):
        if observer not in self.eventControllers: self.eventControllers.append(observer)

    
    ## Remove observer from the list
    #
    # @param observer: ProgressInfoPresentor object
    def detachEventController(self, observer):
        if observer in self.eventControllers: self.eventControllers.remove(observer)


    ## Notify all observers about the new event
    #
    # @param event: LinearSolverEvents event
    def notifyEventControllers(self, event):
        for controller in self.eventControllers:
            controller.handleSolvingEvent(self, event)

class LinearSolverEvents(object):
    STARTING = 100
    COMPLETE = 200
    ITERATION_COMPLETE = 300
    CALC_STARTING = 400
