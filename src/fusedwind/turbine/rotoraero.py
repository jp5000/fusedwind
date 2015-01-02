
import numpy as np
from openmdao.main.api import VariableTree
from openmdao.lib.datatypes.api import Int, Float, Array, List, Str, Enum, Bool, VarTree, Dict

class DistributedLoadsVT(VariableTree):

    s = Array(units='m', desc='locations for distributed loads')
    Fn = Array(units='N/m', desc='force per unit length in normal direction to the blade')
    Ft = Array(units='N/m', desc='force per unit length in tangential direction to the blade')


class DistributedLoadsExtVT(DistributedLoadsVT):

    cn      = Array(units=None, desc='Normal force coefficient along the blade')
    ct      = Array(units=None, desc='Tangential force coefficient along the blade')
    cl      = Array(units=None, desc='Lift force coefficient along the blade')
    cd      = Array(units=None, desc='Drag force coefficient along the blade')
    cm      = Array(units=None, desc='Moment force coefficient along the blade')
    aoa     = Array(units='deg', desc='Angle of attack along the blade')
    lfa     = Array(units='deg', desc='Local flow angle along the blade')
    v_a     = Array(units='m/s', desc='axial velocity along the blade')
    v_t     = Array(units='m/s', desc='tangential velocity along the blade')
    v_r     = Array(units='m/s', desc='radial velocity along the blade')
    lcp     = Array(units=None, desc='Local power coefficient along the blade')
    lct     = Array(units=None, desc='Local power coefficient along the blade')


class RotorLoadsVT(VariableTree):

    T = Float(units='N', desc='thrust')
    Q = Float(units='N*m', desc='torque')
    P = Float(units='W', desc='power')

    CT = Float(units='N', desc='thrust coefficient')
    CQ = Float(units='N*m', desc='torque coefficient')
    CP = Float(units='W', desc='power coefficient')


class RotorLoadsArrayVT(VariableTree):

    wsp = Array(units='m/s', desc='Wind speeds')
    T = Array(units='N', desc='thrust')
    Q = Array(units='N*m', desc='torque')
    P = Array(units='W', desc='power')

    CT = Array(units=None, desc='thrust coefficient')
    CQ = Array(units=None, desc='torque coefficient')
    CP = Array(units=None, desc='power coefficient')


class DistributedLoadsArrayVT(VariableTree):
    """
    Container for a list of blade loads
    """

    loads_array = List(desc='List of arrays of spanwise loads')

    def add_case(self, obj, wsp=None, case_name=None):
        """
        Add a BeamDisplacementsVT

        Specify either wsp or a user specified case name

        Parameters
        ----------
        obj: BeamDisplacementsVT object 
            case to be added
        wsp: float
            optional wind speed
        case_name: str
            custom case name
        """

        if wsp == None and case_name == None:
            raise RuntimeError('Expected either wsp or case_name, got ' 
                % (wsp, case_name))
        if wsp is not None:
            name = 'loads%2.2f' % wsp
        elif case_name is not None:
            name = case_name

        self.add(name, VarTree(obj))
        self.loads_array.append(name)

        return getattr(self, name)


class BeamDisplacementsVT(VariableTree):
    """
    container for beam displacements and rotations
    """
    x = Array(desc='deformed pitch axis edgewise displacement')
    y = Array(desc='deformed pitch axis flapwise displacement')
    z = Array(desc='deformed pitch axis radial displacement')
    rot_x = Array(desc='deformed pitch axis x-rotation')
    rot_y = Array(desc='deformed pitch axis y-rotation')
    rot_z = Array(desc='deformed pitch axis z-rotation')


class BeamDisplacementsArrayVT(VariableTree):
    """
    Container for a series of BeamDisplacementsVT's
    computed for different inflow cases

    Each BeamDisplacementsVT can be added using the add_case method
    """

    disps_array = List(desc='List of names of displacement arrays')
    tip_pos = Array(desc='Tip deflections')
    tip_rot = Array(desc='Tip rotations')

    def add_case(self, obj, wsp=None, case_name=None):
        """
        Add a BeamDisplacementsVT

        Specify either wsp or a user specified case name

        Parameters
        ----------
        obj: BeamDisplacementsVT object 
            case to be added
        wsp: float
            optional wind speed
        case_name: str
            custom case name
        """

        if wsp == None and case_name == None:
            raise RuntimeError('Expected either wsp or case_name, got ' 
                % (wsp, case_name))
        if wsp is not None:
            name = 'loads%2.2f' % wsp
        elif case_name is not None:
            name = case_name

        self.add(name, VarTree(obj))
        self.disps_array.append(name)

        return getattr(self, name)


class LoadVector(VariableTree):
    """
    Point load vector containing forces and moments
    """
    case_id = Str('dlcx.x', desc='Case identifier')
    s = Float(desc='Running length of blade')
    Fx = Float(units='N', desc='Force in x-direction')
    Fy = Float(units='N', desc='Force in y-direction')
    Fz = Float(units='N', desc='Force in z-direction')
    Fres = Float(units='N', desc='Maximum force magnitude')
    Mx = Float(units='N*m', desc='Moment in x-direction')
    My = Float(units='N*m', desc='Moment in y-direction')
    Mz = Float(units='N*m', desc='Moment in z-direction')
    Mres = Float(units='N*m', desc='Moment magnitude')

    def _toarray(self):

        return np.array([self.s, self.Fx, self.Fy, self.Fz, self.Fres,
                                 self.Mx, self.My, self.Mz, self.Mres])

    def _fromarray(self, d):

        self.s = d[0]
        self.Fx = d[1]
        self.Fy = d[2]
        self.Fz = d[3]
        self.Fres = d[4]
        self.Mx = d[5]
        self.My = d[6]
        self.Mz = d[7]
        self.Mres = d[8]


class LoadVectorCaseList(VariableTree):
    """
    List of load vector cases for a given spanwise position
    """
    cases = List(desc='List of load cases')


class LoadVectorArray(VariableTree):
    """
    Load vector case as function of span
    """
    case_id = Str('dlcx.x', desc='Case identifier')
    s = Array(desc='Running length of blade')
    Fx = Array(units='N', desc='Maximum force in x-direction')
    Fy = Array(units='N', desc='Maximum force in y-direction')
    Fz = Array(units='N', desc='Maximum force in z-direction')
    Fres = Array(units='N', desc='Maximum force magnitude')
    Mx = Array(units='N*m', desc='Maximum moment in x-direction')
    My = Array(units='N*m', desc='Maximum moment in y-direction')
    Mz = Array(units='N*m', desc='Maximum moment in z-direction')
    Mres = Array(units='N*m', desc='Maximum moment magnitude')

    def _toarray(self):

        return np.array([self.s, self.Fx, self.Fy, self.Fz, self.Fres,
                                 self.Mx, self.My, self.Mz, self.Mres]).T

    def _fromarray(self, d):

        self.s = d[:, 0]
        self.Fx = d[:, 1]
        self.Fy = d[:, 2]
        self.Fz = d[:, 3]
        self.Fres = d[:, 4]
        self.Mx = d[:, 5]
        self.My = d[:, 6]
        self.Mz = d[:, 7]
        self.Mres = d[:, 8]

    def _interp_s(self, s):

        cn = np.zeros(9)
        arr = self._toarray()
        for i in range(9):
            cn[i] = np.interp(s, self.s, arr[:, i])

        vt = LoadVector()
        vt._fromarray(cn)
        return vt


class LoadVectorArrayCaseList(VariableTree):
    """
    List of load vector cases as function of span
    """
    cases = List(LoadVectorArray, desc='List of load cases')

    def _interp_s(self, s):
        """
        interpolate the case list at a specific location ``s``

        Parameters
        ----------
        s: float
            curve fraction at which to interpolate the data

        Returns
        -------
        lc2d: LoadVectorCaseList
            Variable tree containing list of cases at ``s``
        """

        lc2d = LoadVectorCaseList()
        for case in self.cases:
            lc2d.cases.append(case._interp_s(s))

        return lc2d


