from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.AllChem import AlignMol, EmbedMultipleConfs
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
import numpy as np
import scipy.spatial
from .helpers import Renormalize, SimilarityMetric, psi4Charges, mlCharges, check_hs

# Precomputed coefficients for Gaussian integral calculation
_GAUSS_COEFF_A = np.array(
    [
        [15.90600036, 3.9534831, 17.61453176],
        [3.9534831, 5.21580206, 1.91045387],
        [17.61453176, 1.91045387, 238.75820253],
    ]
)
_GAUSS_COEFF_B = np.array(
    [
        [-0.02495, -0.04539319, -0.00247124],
        [-0.04539319, -0.2513, -0.00258662],
        [-0.00247124, -0.00258662, -0.0013],
    ]
)
_GAUSS_COEFF_A_FLAT = _GAUSS_COEFF_A.flatten()
_GAUSS_COEFF_B_FLAT = _GAUSS_COEFF_B.flatten()


def GetMolProps(
    mol,
    cid,
    charge=None,
    partialCharges="gasteiger",
    basisPsi4="3-21G",
    methodPsi4="scf",
    gridPsi4=1,
):
    """
    Extracts the coordinates, van der Waals radii and charges from a given conformer cid of a molecule mol.
    :param mol: RDKit mol object.
    :param cid: Index of the conformer for 3D coordinates.
    :param charge: List or array of charge. If None or empty, charges are calculated based on the parameter partialCharges.
    :param partialCharges: (optional) Partial charge distribution.
    :param basisPsi4: (optional) Basis set for Psi4 calculation.
    :param methodPsi4: (optional) Method for Psi4 calculation.
    :param gridPsi4: (optional) Integer grid point density for ESP evaluation for Psi4 calculation.
    :return: 2D array of coordinates, 1D array of charges.
    """
    if charge is None:
        charge = []

    # Get actual conformer IDs (may not be sequential if conformers were deleted)
    confs_id = [x.GetId() for x in mol.GetConformers()]
    if cid >= len(confs_id):
        raise ValueError(
            f"Conformer index {cid} out of range (molecule has {len(confs_id)} conformers)"
        )
    actual_conf_id = confs_id[cid]
    coor = mol.GetConformer(actual_conf_id).GetPositions()
    if len(charge) == 0:
        if partialCharges == "gasteiger":
            try:
                charge = np.array(
                    [a.GetDoubleProp("_GasteigerCharge") for a in mol.GetAtoms()]
                )
            except KeyError:
                AllChem.ComputeGasteigerCharges(mol)
                charge = np.array(
                    [a.GetDoubleProp("_GasteigerCharge") for a in mol.GetAtoms()]
                )
        elif partialCharges == "mmff":
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            if mp:
                charge = np.array(
                    [mp.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())]
                )
            else:
                print(
                    "MMFF charges not available for the input molecule, defaulting to Gasteiger charges."
                )
                AllChem.ComputeGasteigerCharges(mol)
                charge = np.array(
                    [a.GetDoubleProp("_GasteigerCharge") for a in mol.GetAtoms()]
                )
        elif partialCharges == "ml":
            charge = np.array(mlCharges([mol])[0])

        elif partialCharges == "resp":
            xyz = Chem.rdmolfiles.MolToXYZBlock(mol, confId=actual_conf_id)
            charge = psi4Charges(xyz, basisPsi4, methodPsi4, gridPsi4)
        else:
            raise ValueError("Unknown partial charge distribution.")
        if charge.shape[0] != coor.shape[0]:
            raise ValueError("Error in partial charge calculation.")
    else:
        charge = np.array(charge, dtype=float).flatten()
        if charge.shape[0] != coor.shape[0]:
            raise ValueError(
                "Dimensions of the supplied charges does not match dimensions of coordinates of molecule"
            )

    return coor, charge


def GetShapeSim(prbMol, refMol, prbCid=-1, refCid=-1):
    """
    Calculates the similarity of the shape between two previously aligned molecules.
    :param prbMol: RDKit mol object of the probe molecule.
    :param refMol: RDKit mol object of the reference molecule.
    :param prbCid: Index of the conformer of the probe molecule to be used for 3D coordinates.
    :param refCid: Index of the conformer of the reference molecule to be used for 3D coordinates.
    :return: Shape score
    """

    return 1 - AllChem.ShapeTanimotoDist(prbMol, refMol, confId1=prbCid, confId2=refCid)


def GetEspSim(
    prbMol,
    refMol,
    prbCid=-1,
    refCid=-1,
    prbCharge=None,
    refCharge=None,
    metric="carbo",
    integrate="gauss",
    partialCharges="gasteiger",
    renormalize=False,
    customrange=None,
    marginMC=10,
    nMC=1,
    basisPsi4="3-21G",
    methodPsi4="scf",
    gridPsi4=1,
    nocheck=False,
    randomseed=2342,
):
    """
    Calculates the similarity of the electrostatic potential around two previously aligned molecules.
    :param prbMol: RDKit mol object of the probe molecule.
    :param refMol: RDKit mol object of the reference molecule.
    :param prbCid: Index of the conformer of the probe molecule to be used for 3D coordinates.
    :param refCid: Index of the conformer of the reference molecule to be used for 3D coordinates.
    :param prbCharge: (optional) List or array of partial charges of the probe molecule. If not given, RDKit Gasteiger Charges are used as default.
    :param refCharge: (optional) List or array of partial charges of the reference molecule. If not given, RDKit Gasteiger Charges are used as default.
    :param metric:  (optional) Similarity metric.
    :param integrate: (optional) Integration method ("gauss" or "mc").
    :param partialCharges: (optional) Partial charge distribution.
    :param renormalize: (optional) Boolean whether to renormalize the similarity score to [0:1].
    :param customrange: (optional) Custom range to renormalize to, supply as tuple or list of two values (lower bound, upper bound).
    :param marginMC: (optional) Margin up to which to integrate (added to coordinates plus/minus their vdW radii) if MC integration is utilized.
    :param nMC: (optional) Number of grid points per 1 Angstrom**3 volume of integration vox if MC integration is utilized.
    :param basisPsi4: (optional) Basis set for Psi4 calculation.
    :param methodPsi4: (optional) Method for Psi4 calculation.
    :param gridPsi4: (optional) Integer grid point density for ESP evaluation for Psi4 calculation.
    :param nocheck: (optional) whether no checks on explicit hydrogens should be run. Speeds up the function, but use wisely.
    :param randomseed: (optional) seed for the random number generator. Only used with the `mc` integration method.
    :return: Similarity score.
    """
    # Handle mutable default arguments
    if prbCharge is None:
        prbCharge = []
    if refCharge is None:
        refCharge = []

    # Validate integrate parameter
    if integrate not in ("gauss", "mc"):
        raise ValueError(
            f"Unknown integration method '{integrate}'. Must be 'gauss' or 'mc'."
        )

    # Check hydrogens
    if not nocheck:
        check_hs(prbMol)
        check_hs(refMol)

    # Set up probe molecule properties:
    prbCoor, prbCharge = GetMolProps(
        prbMol, prbCid, prbCharge, partialCharges, basisPsi4, methodPsi4, gridPsi4
    )
    refCoor, refCharge = GetMolProps(
        refMol, refCid, refCharge, partialCharges, basisPsi4, methodPsi4, gridPsi4
    )

    if integrate == "gauss":
        similarity = GetIntegralsViaGaussians(
            prbCoor, refCoor, prbCharge, refCharge, metric
        )
    else:  # integrate == "mc"
        prbVdw = np.array(
            [
                Chem.GetPeriodicTable().GetRvdw(a.GetAtomicNum())
                for a in prbMol.GetAtoms()
            ]
        ).reshape(-1, 1)
        refVdw = np.array(
            [
                Chem.GetPeriodicTable().GetRvdw(a.GetAtomicNum())
                for a in refMol.GetAtoms()
            ]
        ).reshape(-1, 1)
        similarity = GetIntegralsViaMC(
            prbCoor,
            refCoor,
            prbCharge,
            refCharge,
            prbVdw,
            refVdw,
            metric,
            marginMC,
            nMC,
            randomseed=randomseed,
        )

    if renormalize:
        similarity = Renormalize(similarity, metric, customrange)

    return similarity


def GetIntegralsViaGaussians(
    prbCoor,
    refCoor,
    prbCharge,
    refCharge,
    metric,
):
    """
    Calculates the integral of the overlap between the point charges prbCharge and refCharge at coordinates prbCoor and refCoor via fitting to Gaussian functions and analytic integration.
    :param prbCoor: 2D array of coordinates of the probe molecule.
    :param refCoor: 2D array of coordinates of the reference molecule.
    :param prbCharge: 1D array of partial charges of the probe molecule.
    :param refCharge: 1D array of partial charges of the reference molecule.
    :param metric: Metric of similarity score.
    :return: Similarity of the overlap integrals.
    """

    distPrbPrb = scipy.spatial.distance.cdist(prbCoor, prbCoor)
    distPrbRef = scipy.spatial.distance.cdist(prbCoor, refCoor)
    distRefRef = scipy.spatial.distance.cdist(refCoor, refCoor)

    intPrbPrb = GaussInt(distPrbPrb, prbCharge, prbCharge)
    intPrbRef = GaussInt(distPrbRef, prbCharge, refCharge)
    intRefRef = GaussInt(distRefRef, refCharge, refCharge)

    similarity = SimilarityMetric(intPrbPrb, intRefRef, intPrbRef, metric)
    return similarity


def GaussInt(
    dist,
    charge1,
    charge2,
):
    """Calculates the analytic Gaussian integrals.
    :param dist: Distance matrix.
    :param charge1: 1D array of partial charges of first molecule.
    :param charge2: 1D array of partial charges of second molecule.
    :return: Analytic overlap integral.
    """
    dist_sq = (dist**2).flatten()
    charges = (
        charge1[:, None] * charge2
    ).flatten()  # pairwise products of atomic charges, flattened
    return (
        (
            _GAUSS_COEFF_A_FLAT[:, None]
            * np.exp(dist_sq * _GAUSS_COEFF_B_FLAT[:, None])
        ).sum(0)
        * charges
    ).sum()


def GetIntegralsViaMC(
    prbCoor,
    refCoor,
    prbCharge,
    refCharge,
    prbVdw,
    refVdw,
    metric,
    marginMC=10,
    nMC=1,
    randomseed=2342,
):
    """
    Calculates the integral of the overlap between the point charges prbCharge and refCharge
    at coordinates prbCoor and refCoor via Monte Carlo numeric integration.
    :param prbCoor: 2D array of coordinates of the probe molecule.
    :param refCoor: 2D array of coordinates of the reference molecule.
    :param prbCharge: 1D array of partial charges of the probe molecule.
    :param refCharge: 1D array of partial charges of the reference molecule.
    :param metric: Metric of similarity score.
    :param marginMC: (optional) Margin up to which to integrate (added to coordinates plus/minus their vdW radii).
    :param nMC: (optional) Number of grid points per 1 Angstrom**3 volume of integration vox.
    :param randomseed: (optional) seed for the random number generator
    :return: Similarity of the overlap integrals.
    """
    rng = np.random.default_rng(randomseed)
    margin = marginMC
    allCoor = np.concatenate((prbCoor, refCoor))
    allVdw = np.concatenate((prbVdw, refVdw)).flatten()

    minValues = np.min(allCoor - allVdw[:, None] - margin, axis=0)
    maxValues = np.max(allCoor + allVdw[:, None] + margin, axis=0)  # Fixed: was np.min
    boxvolume = np.prod(maxValues - minValues)

    # Handle edge case where box volume is zero or negative
    if boxvolume <= 0:
        raise ValueError(
            "Invalid bounding box for Monte Carlo integration (zero or negative volume)"
        )

    N = max(1, int(boxvolume * nMC))  # Ensure at least 1 sample point

    # Generate all random points at once (vectorized)
    points = rng.uniform(minValues, maxValues, size=(N, 3))

    # Compute distances from all points to all atoms (vectorized)
    distPrb = scipy.spatial.distance.cdist(points, prbCoor)  # Shape: (N, lenPrb)
    distRef = scipy.spatial.distance.cdist(points, refCoor)  # Shape: (N, lenRef)

    # Compute minimum distance to vdW surface for each point
    distAll = np.concatenate((distPrb, distRef), axis=1)  # Shape: (N, lenPrb + lenRef)
    distMinVdw = distAll - allVdw  # Broadcasting: (N, lenAll)
    minDistPerPoint = np.min(distMinVdw, axis=1)  # Shape: (N,)

    # Filter points within valid margin (0 < minDist <= margin)
    valid_mask = (minDistPerPoint > 0) & (minDistPerPoint <= margin)
    nInMargin = np.sum(valid_mask)

    if nInMargin == 0:
        # No valid points - return zero similarity
        return SimilarityMetric(0.0, 0.0, 0.0, metric)

    # Extract valid distances
    distPrb_valid = distPrb[valid_mask]  # Shape: (nInMargin, lenPrb)
    distRef_valid = distRef[valid_mask]  # Shape: (nInMargin, lenRef)

    # Compute electrostatic potential contributions (vectorized)
    # fPrb[i] = sum(prbCharge[j] / distPrb[i,j] for j in range(lenPrb))
    fPrb = np.sum(prbCharge / distPrb_valid, axis=1)  # Shape: (nInMargin,)
    fRef = np.sum(refCharge / distRef_valid, axis=1)  # Shape: (nInMargin,)

    # Compute integrals
    intPrbPrb = np.sum(fPrb * fPrb)
    intPrbRef = np.sum(fPrb * fRef)
    intRefRef = np.sum(fRef * fRef)

    # Apply Monte Carlo scaling factor
    factor = float(nInMargin) / N * boxvolume / N
    intPrbPrb *= factor
    intPrbRef *= factor
    intRefRef *= factor

    similarity = SimilarityMetric(intPrbPrb, intRefRef, intPrbRef, metric)

    return similarity


def ConstrainedEmbedMultipleConfs(
    mol,
    core,
    numConfs=10,
    useTethers=True,
    coreConfId=-1,
    randomSeed=2342,
    getForceField=UFFGetMoleculeForceField,
    **kwargs,
):
    """
    Function to obtain multiple constrained embeddings per molecule. This was taken as is from:
    from https://github.com/rdkit/rdkit/issues/3266
    :param mol: RDKit molecule object to be embedded.
    :param core: RDKit molecule object of the core used as constrained. Needs to hold at least one conformer coordinates.
    :param numCons: Number of conformations to create
    :param useTethers: (optional) boolean whether to pull embedded atoms to core coordinates, see rdkit.Chem.AllChem.ConstrainedEmbed
    :param coreConfId: (optional) id of the core conformation to use
    :param randomSeed: (optional) seed for the random number generator
    :param getForceField: (optional) force field to use for the optimization of molecules
    :return: RDKit molecule object containing the embedded conformations.
    """

    match = mol.GetSubstructMatch(core)
    if not match:
        raise ValueError("molecule doesn't match the core")
    coordMap = {}
    coreConf = core.GetConformer(coreConfId)
    for i, idxI in enumerate(match):
        corePtI = coreConf.GetAtomPosition(i)
        coordMap[idxI] = corePtI

    cids = EmbedMultipleConfs(
        mol, numConfs=numConfs, coordMap=coordMap, randomSeed=randomSeed, **kwargs
    )
    cids = list(cids)
    if len(cids) == 0:
        raise ValueError("Could not embed molecule.")

    algMap = [(j, i) for i, j in enumerate(match)]

    if not useTethers:
        # clean up the conformation
        for cid in cids:
            ff = getForceField(mol, confId=cid)
            for i, idxI in enumerate(match):
                for j in range(i + 1, len(match)):
                    idxJ = match[j]
                    d = coordMap[idxI].Distance(coordMap[idxJ])
                    ff.AddDistanceConstraint(idxI, idxJ, d, d, 100.0)
            ff.Initialize()
            n = 4
            more = ff.Minimize()
            while more and n:
                more = ff.Minimize()
                n -= 1
            # rotate the embedded conformation onto the core:
            AlignMol(mol, core, atomMap=algMap)
    else:
        # rotate the embedded conformation onto the core:
        for cid in cids:
            AlignMol(mol, core, prbCid=cid, atomMap=algMap)
            ff = getForceField(mol, confId=cid)
            conf = core.GetConformer()
            for i in range(core.GetNumAtoms()):
                p = conf.GetAtomPosition(i)
                pIdx = ff.AddExtraPoint(p.x, p.y, p.z, fixed=True) - 1
                ff.AddDistanceConstraint(pIdx, match[i], 0, 0, 100.0)
            ff.Initialize()
            n = 4
            more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
            while more and n:
                more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
                n -= 1
            # realign
            AlignMol(mol, core, prbCid=cid, atomMap=algMap)
    return mol


def EmbedAlignConstrainedScore(
    prbMol,
    refMols,
    core,
    prbNumConfs=10,
    refNumConfs=10,
    prbCharge=None,
    refCharges=None,
    metric="carbo",
    integrate="gauss",
    partialCharges="gasteiger",
    renormalize=False,
    customrange=None,
    marginMC=10,
    nMC=1,
    basisPsi4="3-21G",
    methodPsi4="scf",
    gridPsi4=1,
    getBestESP=False,
    randomseed=2342,
):
    """Calculates a constrained alignment based on a common pattern in the input molecules. Caution: Will fail if the pattern does not match.
    Calculates a shape and electrostatic potential similarity of the best alignment.

    :param prbMol: RDKit molecule for which shape and electrostatic similarities are calculated.
    :param refMol: RDKit molecule or list of RDKit molecules serving as references.
    :param core: Common pattern for the constrained embedding as embedded RDKit molecule
    :param prbNumConfs: Number of conformers to create for the probe molecule. A higher number creates better alignments but slows down the algorithm.
    :param refNumConfs: Number of conformers to create for each reference molecule. A higher number creates better alignments but slows down the algorithm.
    :param prbCharge: (optional) List or array of partial charges of the probe molecule. If not given, RDKit Gasteiger Charges are used as default.
    :param refCharge: (optional) List of list or 2D array of partial charges of the reference molecules. If not given, RDKit Gasteiger Charges are used as default.
    :param metric:  (optional) Similarity metric.
    :param integrate: (optional) Integration method.
    :param partialCharges: (optional) Partial charge distribution.
    :param renormalize: (optional) Boolean whether to renormalize the similarity score to [0:1].
    :param customrange: (optional) Custom range to renormalize to, supply as tuple or list of two values (lower bound, upper bound).
    :param marginMC: (optional) Margin up to which to integrate (added to coordinates plus/minus their vdW radii) if MC integration is utilized.
    :param nMC: (optional) Number of grid points per 1 Angstrom**3 volume of integration vox if MC integration is utilized.
    :param basisPsi4: (optional) Basis set for Psi4 calculation.
    :param methodPsi4: (optional) Method for Psi4 calculation.
    :param gridPsi4: (optional) Integer grid point density for ESP evaluation for Psi4 calculation.
    :param getBestESP: (optional) Whether to select best alignment via ESP instead of shape.
    :param randomseed: (optional) seed for the random number generator
    :return: shape similarity and ESP similarity.
    """
    # Handle mutable default arguments
    if prbCharge is None:
        prbCharge = []
    if refCharges is None:
        refCharges = []

    if not isinstance(refMols, list):
        refMols = [refMols]

    if refCharges == []:
        refCharges = [[]] * len(refMols)

    prbMol = ConstrainedEmbedMultipleConfs(
        prbMol, core, numConfs=prbNumConfs, randomSeed=randomseed
    )
    for refMol in refMols:
        refMol = ConstrainedEmbedMultipleConfs(
            refMol, core, numConfs=refNumConfs, randomSeed=randomseed
        )

    # Get actual number of conformers generated
    actualPrbNumConfs = prbMol.GetNumConformers()

    prbMatch = prbMol.GetSubstructMatch(core)
    allShapeSim = []
    allEspSim = []

    if not getBestESP:
        for idx, refMol in enumerate(refMols):
            actualRefNumConfs = refMol.GetNumConformers()
            if actualRefNumConfs == 0:
                allShapeSim.append(0)
                allEspSim.append(0)
                continue

            shapeSim = 0
            prbBestConf = 0
            refBestConf = 0
            foundValidAlignment = False
            refMatch = refMol.GetSubstructMatch(core)
            for i in range(actualRefNumConfs):
                for j in range(actualPrbNumConfs):
                    try:
                        AllChem.AlignMol(
                            prbMol,
                            refMol,
                            atomMap=list(zip(prbMatch, refMatch)),
                            prbCid=j,
                            refCid=i,
                        )
                        shape = GetShapeSim(prbMol, refMol, j, i)
                        foundValidAlignment = True
                        if shape > shapeSim:
                            shapeSim = shape
                            prbBestConf = j
                            refBestConf = i
                    except ValueError:
                        # Skip conformer pairs with invalid conformer IDs
                        continue

            # If no valid alignment was found, return zeros for this pair
            if not foundValidAlignment:
                allShapeSim.append(0)
                allEspSim.append(0)
                continue

            # Go back to best alignment
            try:
                AllChem.AlignMol(
                    prbMol,
                    refMol,
                    atomMap=list(zip(prbMatch, refMatch)),
                    prbCid=prbBestConf,
                    refCid=refBestConf,
                )
            except ValueError:
                # If best alignment fails, return zeros for this pair
                allShapeSim.append(0)
                allEspSim.append(0)
                continue

            espSim = GetEspSim(
                prbMol,
                refMol,
                prbBestConf,
                refBestConf,
                prbCharge,
                refCharges[idx],
                metric,
                integrate,
                partialCharges,
                renormalize,
                customrange,
                marginMC,
                nMC,
                basisPsi4,
                methodPsi4,
                gridPsi4,
                randomseed=randomseed,
            )
            allShapeSim.append(shapeSim)
            allEspSim.append(espSim)
    else:
        for idx, refMol in enumerate(refMols):
            actualRefNumConfs = refMol.GetNumConformers()
            if actualRefNumConfs == 0:
                allShapeSim.append(0)
                allEspSim.append(0)
                continue

            espSim = 0
            shapeSim = 0
            refMatch = refMol.GetSubstructMatch(core)
            for i in range(actualRefNumConfs):
                for j in range(actualPrbNumConfs):
                    try:
                        AllChem.AlignMol(
                            prbMol,
                            refMol,
                            atomMap=list(zip(prbMatch, refMatch)),
                            prbCid=j,
                            refCid=i,
                        )
                        score = GetEspSim(
                            prbMol,
                            refMol,
                            j,
                            i,
                            prbCharge,
                            refCharges[idx],
                            metric,
                            integrate,
                            partialCharges,
                            renormalize,
                            customrange,
                            marginMC,
                            nMC,
                            basisPsi4,
                            methodPsi4,
                            gridPsi4,
                            randomseed=randomseed,
                        )
                        if score > espSim:
                            espSim = score
                        shape = GetShapeSim(prbMol, refMol, j, i)
                        if shape > shapeSim:
                            shapeSim = shape
                    except ValueError:
                        # Skip conformer pairs with invalid conformer IDs
                        continue
            allShapeSim.append(shapeSim)
            allEspSim.append(espSim)

    return allShapeSim, allEspSim


def EmbedAlignScore(
    prbMol,
    refMols,
    prbNumConfs=10,
    refNumConfs=10,
    prbCharge=None,
    refCharges=None,
    metric="carbo",
    integrate="gauss",
    partialCharges="gasteiger",
    renormalize=False,
    customrange=None,
    marginMC=10,
    nMC=1,
    basisPsi4="3-21G",
    methodPsi4="scf",
    gridPsi4=1,
    getBestESP=False,
    randomseed=2342,
):
    """Calculates a general alignment in the input molecules.
    Calculates a shape and electrostatic potential similarity of the best alignment.

    :param prbMol: RDKit molecule for which shape and electrostatic similarities are calculated.
    :param refMol: RDKit molecule or list of RDKit molecules serving as references.
    :param prbNumConfs: Number of conformers to create for the probe molecule. A higher number creates better alignments but slows down the algorithm.
    :param refNumConfs: Number of conformers to create for each reference molecule. A higher number creates better alignments but slows down the algorithm.
    :param prbCharge: (optional) List or array of partial charges of the probe molecule. If not given, RDKit Gasteiger Charges are used as default.
    :param refCharge: (optional) List of list or 2D array of partial charges of the reference molecules. If not given, RDKit Gasteiger Charges are used as default.
    :param metric:  (optional) Similarity metric.
    :param integrate: (optional) Integration method.
    :param partialCharges: (optional) Partial charge distribution.
    :param renormalize: (optional) Boolean whether to renormalize the similarity score to [0:1].
    :param customrange: (optional) Custom range to renormalize to, supply as tuple or list of two values (lower bound, upper bound).
    :param marginMC: (optional) Margin up to which to integrate (added to coordinates plus/minus their vdW radii) if MC integration is utilized.
    :param nMC: (optional) Number of grid points per 1 Angstrom**3 volume of integration vox if MC integration is utilized.
    :param basisPsi4: (optional) Basis set for Psi4 calculation.
    :param methodPsi4: (optional) Method for Psi4 calculation.
    :param gridPsi4: (optional) Integer grid point density for ESP evaluation for Psi4 calculation.
    :param getBestESP: Whether to select best alignment via ESP instead of shape.
    :param randomseed: (optional) seed for the random number generator
    :return: shape similarity and ESP similarity.
    """
    # Handle mutable default arguments
    if prbCharge is None:
        prbCharge = []
    if refCharges is None:
        refCharges = []

    if not isinstance(refMols, list):
        refMols = [refMols]

    if refCharges == []:
        refCharges = [[]] * len(refMols)

    AllChem.EmbedMultipleConfs(prbMol, prbNumConfs, randomSeed=randomseed)
    for refMol in refMols:
        AllChem.EmbedMultipleConfs(refMol, refNumConfs, randomSeed=randomseed)

    # Get actual number of conformers generated (may be less than requested)
    actualPrbNumConfs = prbMol.GetNumConformers()
    if actualPrbNumConfs == 0:
        raise ValueError("Failed to generate any conformers for probe molecule")

    prbCrippen = rdMolDescriptors._CalcCrippenContribs(prbMol)

    allShapeSim = []
    allEspSim = []

    if not getBestESP:
        for idx, refMol in enumerate(refMols):
            actualRefNumConfs = refMol.GetNumConformers()
            if actualRefNumConfs == 0:
                # No conformers for this reference - append zero similarity
                allShapeSim.append(0)
                allEspSim.append(0)
                continue

            shapeSim = 0
            prbBestConf = 0
            refBestConf = 0
            foundValidAlignment = False
            refCrippen = rdMolDescriptors._CalcCrippenContribs(refMol)
            for i in range(actualRefNumConfs):
                for j in range(actualPrbNumConfs):
                    try:
                        alignment = rdMolAlign.GetCrippenO3A(
                            prbMol, refMol, prbCrippen, refCrippen, j, i
                        )
                        alignment.Align()
                        shape = GetShapeSim(prbMol, refMol, j, i)
                        foundValidAlignment = True
                        if shape > shapeSim:
                            shapeSim = shape
                            prbBestConf = j
                            refBestConf = i
                    except ValueError:
                        # Skip conformer pairs with invalid conformer IDs
                        continue

            # If no valid alignment was found, return zeros for this pair
            if not foundValidAlignment:
                allShapeSim.append(0)
                allEspSim.append(0)
                continue

            # Go back to best alignment
            try:
                alignment = rdMolAlign.GetCrippenO3A(
                    prbMol, refMol, prbCrippen, refCrippen, prbBestConf, refBestConf
                )
                alignment.Align()
            except ValueError:
                # If best alignment fails, return zeros for this pair
                allShapeSim.append(0)
                allEspSim.append(0)
                continue

            espSim = GetEspSim(
                prbMol,
                refMol,
                prbBestConf,
                refBestConf,
                prbCharge,
                refCharges[idx],
                metric,
                integrate,
                partialCharges,
                renormalize,
                customrange,
                marginMC,
                nMC,
                basisPsi4,
                methodPsi4,
                gridPsi4,
                randomseed=randomseed,
            )
            allShapeSim.append(shapeSim)
            allEspSim.append(espSim)
    else:
        for idx, refMol in enumerate(refMols):
            actualRefNumConfs = refMol.GetNumConformers()
            if actualRefNumConfs == 0:
                # No conformers for this reference - append zero similarity
                allShapeSim.append(0)
                allEspSim.append(0)
                continue

            espSim = 0
            shapeSim = 0
            prbBestConf = 0
            refBestConf = 0
            refCrippen = rdMolDescriptors._CalcCrippenContribs(refMol)
            for i in range(actualRefNumConfs):
                for j in range(actualPrbNumConfs):
                    try:
                        alignment = rdMolAlign.GetCrippenO3A(
                            prbMol, refMol, prbCrippen, refCrippen, j, i
                        )
                        alignment.Align()
                        score = GetEspSim(
                            prbMol,
                            refMol,
                            j,
                            i,
                            prbCharge,
                            refCharges[idx],
                            metric,
                            integrate,
                            partialCharges,
                            renormalize,
                            customrange,
                            marginMC,
                            nMC,
                            basisPsi4,
                            methodPsi4,
                            gridPsi4,
                            randomseed=randomseed,
                        )
                        if score > espSim:
                            espSim = score
                        shape = GetShapeSim(prbMol, refMol, j, i)
                        if shape > shapeSim:
                            shapeSim = shape
                    except ValueError:
                        # Skip conformer pairs with invalid conformer IDs
                        continue
            allShapeSim.append(shapeSim)
            allEspSim.append(espSim)

    return allShapeSim, allEspSim
