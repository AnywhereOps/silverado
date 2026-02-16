// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title Settlement — Payment settlement after quantum work finalization
 * @notice Handles the economics of the quantum simulation marketplace:
 *         - Problem submitters deposit reward pools
 *         - Solvers earn proportional to their contribution
 *         - Verifiers earn fees for honest verification
 *         - Protocol earns a small fee
 *
 * Payment is triggered when a submission is finalized in the Rollup contract.
 */
contract Settlement is ReentrancyGuard {

    IERC20 public token;
    address public rollup;              // Authorized caller
    address public protocolFeeRecipient;

    uint256 public constant PROTOCOL_FEE_BPS = 200;     // 2% protocol fee
    uint256 public constant VERIFIER_POOL_BPS = 500;     // 5% to verifier pool
    uint256 public constant SOLVER_SHARE_BPS = 9300;     // 93% to solvers

    struct RewardPool {
        address submitter;
        bytes32 problemHash;
        uint256 totalReward;
        uint256 deadline;
        bool settled;
        uint256 bestSubmissionId;      // Winning submission
        address bestSolver;
    }

    struct SolverContribution {
        address solver;
        uint256 totalSamples;
        int256 bestEnergy;
        bool claimed;
    }

    uint256 public poolCount;
    mapping(uint256 => RewardPool) public pools;
    mapping(uint256 => SolverContribution[]) public contributions;  // poolId → contributions
    mapping(uint256 => uint256) public verifierPoolBalance;  // poolId → accumulated verifier fees

    event PoolCreated(uint256 indexed poolId, bytes32 problemHash, uint256 reward, uint256 deadline);
    event ContributionRecorded(uint256 indexed poolId, address indexed solver, uint256 samples, int256 energy);
    event PoolSettled(uint256 indexed poolId, address indexed winner, uint256 reward);
    event VerifierPaid(uint256 indexed poolId, address indexed verifier, uint256 amount);

    constructor(address _token, address _rollup, address _feeRecipient) {
        token = IERC20(_token);
        rollup = _rollup;
        protocolFeeRecipient = _feeRecipient;
    }

    /// @notice Create a reward pool for a problem
    function createPool(
        bytes32 problemHash,
        uint256 totalReward,
        uint256 deadline
    ) external returns (uint256 poolId) {
        require(totalReward > 0, "No reward");
        require(deadline > block.timestamp, "Deadline must be future");

        token.transferFrom(msg.sender, address(this), totalReward);

        poolId = poolCount++;
        pools[poolId] = RewardPool({
            submitter: msg.sender,
            problemHash: problemHash,
            totalReward: totalReward,
            deadline: deadline,
            settled: false,
            bestSubmissionId: 0,
            bestSolver: address(0)
        });

        emit PoolCreated(poolId, problemHash, totalReward, deadline);
    }

    /// @notice Record a solver's contribution to a pool
    function recordContribution(
        uint256 poolId,
        address solver,
        uint256 totalSamples,
        int256 bestEnergy
    ) external {
        // In production, this would only be callable by the Rollup contract
        RewardPool storage pool = pools[poolId];
        require(!pool.settled, "Already settled");

        contributions[poolId].push(SolverContribution({
            solver: solver,
            totalSamples: totalSamples,
            bestEnergy: bestEnergy,
            claimed: false
        }));

        // Track best solver
        if (pool.bestSolver == address(0) || bestEnergy > int256(0)) {
            // Simple: best energy wins
            bool isBest = true;
            for (uint256 i = 0; i < contributions[poolId].length - 1; i++) {
                if (contributions[poolId][i].bestEnergy > bestEnergy) {
                    isBest = false;
                    break;
                }
            }
            if (isBest) {
                pool.bestSolver = solver;
            }
        }

        emit ContributionRecorded(poolId, solver, totalSamples, bestEnergy);
    }

    /// @notice Settle a pool — distribute rewards after deadline
    function settle(uint256 poolId) external nonReentrant {
        RewardPool storage pool = pools[poolId];
        require(!pool.settled, "Already settled");
        require(block.timestamp >= pool.deadline, "Deadline not reached");

        pool.settled = true;

        SolverContribution[] storage contribs = contributions[poolId];
        if (contribs.length == 0) {
            // No contributions — refund submitter
            token.transfer(pool.submitter, pool.totalReward);
            return;
        }

        // Calculate fee splits
        uint256 protocolFee = (pool.totalReward * PROTOCOL_FEE_BPS) / 10000;
        uint256 verifierFee = (pool.totalReward * VERIFIER_POOL_BPS) / 10000;
        uint256 solverPool = pool.totalReward - protocolFee - verifierFee;

        // Protocol fee
        token.transfer(protocolFeeRecipient, protocolFee);

        // Verifier pool (to be claimed later)
        verifierPoolBalance[poolId] = verifierFee;

        // Find winner (best energy)
        int256 bestEnergy = type(int256).min;
        address winner = address(0);
        uint256 totalSamples = 0;

        for (uint256 i = 0; i < contribs.length; i++) {
            totalSamples += contribs[i].totalSamples;
            if (contribs[i].bestEnergy > bestEnergy) {
                bestEnergy = contribs[i].bestEnergy;
                winner = contribs[i].solver;
            }
        }

        // Distribution: 70% to winner, 30% proportional to samples
        uint256 winnerBonus = (solverPool * 70) / 100;
        uint256 participationPool = solverPool - winnerBonus;

        // Pay winner
        if (winner != address(0)) {
            token.transfer(winner, winnerBonus);
        }

        // Pay proportional to samples contributed
        if (totalSamples > 0) {
            for (uint256 i = 0; i < contribs.length; i++) {
                uint256 share = (participationPool * contribs[i].totalSamples) / totalSamples;
                if (share > 0) {
                    token.transfer(contribs[i].solver, share);
                    contribs[i].claimed = true;
                }
            }
        }

        emit PoolSettled(poolId, winner, solverPool);
    }

    /// @notice Claim verifier fee from a settled pool
    function claimVerifierFee(uint256 poolId, address verifier, uint256 amount) external {
        // In production: verify the caller is an authorized verifier
        require(amount <= verifierPoolBalance[poolId], "Insufficient balance");
        verifierPoolBalance[poolId] -= amount;
        token.transfer(verifier, amount);
        emit VerifierPaid(poolId, verifier, amount);
    }

    // ── View functions ─────────────────────────────────────────────

    function getPool(uint256 poolId) external view returns (RewardPool memory) {
        return pools[poolId];
    }

    function getContributionCount(uint256 poolId) external view returns (uint256) {
        return contributions[poolId].length;
    }
}
