// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title QuantumProblem — Problem submission and solution marketplace
 * @notice Anyone can submit an optimization problem with a reward bounty.
 *         Solvers compete to find the best solution. Verified solutions
 *         earn the reward.
 *
 * Flow:
 *   1. Submitter creates problem with reward deposit
 *   2. Solvers submit solutions (off-chain compute, on-chain proof)
 *   3. Verifiers spot-check solutions
 *   4. Best verified solution before deadline wins the reward
 */
contract QuantumProblem is ReentrancyGuard {

    IERC20 public token;

    enum ProblemStatus { Open, Solved, Expired, Cancelled }

    struct Problem {
        address submitter;
        bytes32 hamiltonianHash;     // Keccak of the serialized Hamiltonian
        string hamiltonianURI;       // IPFS/Arweave URI to full problem data
        uint16 nQubits;
        uint256 reward;              // QC tokens locked as bounty
        uint256 targetEnergy;        // Minimum energy threshold (scaled by 1e18)
        uint256 deadline;
        ProblemStatus status;
        address bestSolver;
        int256 bestEnergy;           // Best solution energy found so far
        bytes32 bestProofHash;       // Hash of the best solution proof
    }

    struct Solution {
        address solver;
        int256 energy;               // Claimed energy of the solution
        bytes32 proofHash;           // Hash of measurement samples + bitstring
        string proofURI;             // IPFS URI to full proof data
        uint256 timestamp;
        bool verified;
        bool rejected;
    }

    uint256 public problemCount;
    mapping(uint256 => Problem) public problems;
    mapping(uint256 => Solution[]) public solutions;

    // Fees
    uint256 public constant SUBMISSION_FEE_BPS = 100; // 1% fee on rewards
    uint256 public constant VERIFIER_SHARE_BPS = 500; // 5% of reward goes to verifiers
    address public feeCollector;

    event ProblemCreated(uint256 indexed problemId, address indexed submitter, uint256 reward, uint16 nQubits);
    event SolutionSubmitted(uint256 indexed problemId, uint256 solutionIndex, address indexed solver, int256 energy);
    event SolutionVerified(uint256 indexed problemId, uint256 solutionIndex, bool accepted);
    event ProblemSolved(uint256 indexed problemId, address indexed solver, int256 energy, uint256 reward);
    event ProblemExpired(uint256 indexed problemId);
    event ProblemCancelled(uint256 indexed problemId);

    constructor(address _token, address _feeCollector) {
        token = IERC20(_token);
        feeCollector = _feeCollector;
    }

    /// @notice Submit a new optimization problem with a reward bounty
    function createProblem(
        bytes32 hamiltonianHash,
        string calldata hamiltonianURI,
        uint16 nQubits,
        uint256 reward,
        uint256 targetEnergy,
        uint256 deadline
    ) external returns (uint256 problemId) {
        require(reward > 0, "Reward must be > 0");
        require(deadline > block.timestamp, "Deadline must be in the future");
        require(nQubits > 0 && nQubits <= 1000, "Invalid qubit count");

        // Transfer reward tokens from submitter
        uint256 fee = (reward * SUBMISSION_FEE_BPS) / 10000;
        token.transferFrom(msg.sender, address(this), reward);
        if (fee > 0) {
            token.transferFrom(msg.sender, feeCollector, fee);
        }

        problemId = problemCount++;
        problems[problemId] = Problem({
            submitter: msg.sender,
            hamiltonianHash: hamiltonianHash,
            hamiltonianURI: hamiltonianURI,
            nQubits: nQubits,
            reward: reward,
            targetEnergy: targetEnergy,
            deadline: deadline,
            status: ProblemStatus.Open,
            bestSolver: address(0),
            bestEnergy: type(int256).min,
            bestProofHash: bytes32(0)
        });

        emit ProblemCreated(problemId, msg.sender, reward, nQubits);
    }

    /// @notice Submit a solution to an open problem
    function submitSolution(
        uint256 problemId,
        int256 energy,
        bytes32 proofHash,
        string calldata proofURI
    ) external {
        Problem storage p = problems[problemId];
        require(p.status == ProblemStatus.Open, "Problem not open");
        require(block.timestamp < p.deadline, "Deadline passed");

        uint256 solIndex = solutions[problemId].length;
        solutions[problemId].push(Solution({
            solver: msg.sender,
            energy: energy,
            proofHash: proofHash,
            proofURI: proofURI,
            timestamp: block.timestamp,
            verified: false,
            rejected: false
        }));

        // Update best solution if this one is better
        if (energy > p.bestEnergy) {
            p.bestEnergy = energy;
            p.bestSolver = msg.sender;
            p.bestProofHash = proofHash;
        }

        emit SolutionSubmitted(problemId, solIndex, msg.sender, energy);
    }

    /// @notice Verify a submitted solution (called by authorized verifiers)
    function verifySolution(
        uint256 problemId,
        uint256 solutionIndex,
        bool accepted
    ) external {
        // In production, this would be restricted to staked verifiers
        // and use on-chain verification or ZK proofs
        Solution storage sol = solutions[problemId][solutionIndex];
        require(!sol.verified && !sol.rejected, "Already processed");

        if (accepted) {
            sol.verified = true;
        } else {
            sol.rejected = true;
            // If rejected solution was the best, find next best
            Problem storage p = problems[problemId];
            if (sol.solver == p.bestSolver) {
                _recalculateBest(problemId);
            }
        }

        emit SolutionVerified(problemId, solutionIndex, accepted);
    }

    /// @notice Finalize a problem after deadline. Pays the best verified solver.
    function finalize(uint256 problemId) external nonReentrant {
        Problem storage p = problems[problemId];
        require(p.status == ProblemStatus.Open, "Not open");
        require(block.timestamp >= p.deadline, "Deadline not reached");

        // Find best verified solution
        address winner = address(0);
        int256 bestEnergy = type(int256).min;

        Solution[] storage sols = solutions[problemId];
        for (uint256 i = 0; i < sols.length; i++) {
            if (sols[i].verified && !sols[i].rejected && sols[i].energy > bestEnergy) {
                bestEnergy = sols[i].energy;
                winner = sols[i].solver;
            }
        }

        if (winner != address(0) && bestEnergy >= int256(p.targetEnergy)) {
            // Pay the winner
            uint256 verifierShare = (p.reward * VERIFIER_SHARE_BPS) / 10000;
            uint256 solverReward = p.reward - verifierShare;

            token.transfer(winner, solverReward);
            // Verifier share stays in contract for verifier claims
            p.status = ProblemStatus.Solved;
            emit ProblemSolved(problemId, winner, bestEnergy, solverReward);
        } else {
            // No valid solution — refund submitter
            token.transfer(p.submitter, p.reward);
            p.status = ProblemStatus.Expired;
            emit ProblemExpired(problemId);
        }
    }

    /// @notice Cancel a problem before any solutions are submitted
    function cancel(uint256 problemId) external nonReentrant {
        Problem storage p = problems[problemId];
        require(msg.sender == p.submitter, "Not submitter");
        require(p.status == ProblemStatus.Open, "Not open");
        require(solutions[problemId].length == 0, "Solutions already submitted");

        token.transfer(p.submitter, p.reward);
        p.status = ProblemStatus.Cancelled;
        emit ProblemCancelled(problemId);
    }

    function _recalculateBest(uint256 problemId) internal {
        Problem storage p = problems[problemId];
        p.bestEnergy = type(int256).min;
        p.bestSolver = address(0);

        Solution[] storage sols = solutions[problemId];
        for (uint256 i = 0; i < sols.length; i++) {
            if (!sols[i].rejected && sols[i].energy > p.bestEnergy) {
                p.bestEnergy = sols[i].energy;
                p.bestSolver = sols[i].solver;
            }
        }
    }

    function getSolutionCount(uint256 problemId) external view returns (uint256) {
        return solutions[problemId].length;
    }
}
