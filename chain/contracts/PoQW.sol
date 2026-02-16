// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title PoQW — Proof of Quantum Work
 * @notice Consensus mechanism that replaces SHA-256 mining with useful
 *         quantum simulation.
 *
 * Flow:
 *   1. Miners select a problem from the pool (or default benchmark)
 *   2. Mine by running match-strike cycles off-chain
 *   3. Submit proof: best solution + sample subset for verification
 *   4. Verifiers spot-check a random subset of claimed circuits
 *   5. If verified, miner earns block reward + problem bounty
 *
 * The key asymmetry: producing 10,000 samples is expensive.
 * Verifying 50 of them is cheap.
 */
contract PoQW is ReentrancyGuard {

    // ── Interfaces ─────────────────────────────────────────────────

    interface IQCToken {
        function mintMiningReward(address to, uint256 amount) external;
        function currentMiningReward() external view returns (uint256);
    }

    interface IStaking {
        function isActiveSolver(address addr) external view returns (bool);
        function isActiveVerifier(address addr) external view returns (bool);
        function recordTaskCompletion(address staker) external;
        function slash(address staker, string calldata reason) external;
    }

    IQCToken public token;
    IStaking public staking;

    // ── Block structure ────────────────────────────────────────────

    struct QuantumBlock {
        uint256 blockNumber;
        address miner;
        uint256 problemId;           // Which problem was solved (0 = benchmark)
        bytes32 solutionHash;        // Hash of the best bitstring + cut value
        int256 bestEnergy;           // Best solution energy
        uint256 totalSamples;        // Total match-strike samples computed
        bytes32 sampleMerkleRoot;    // Merkle root of submitted samples (for verification)
        uint256 timestamp;
        bool verified;
        uint256 reward;
    }

    struct VerificationChallenge {
        uint256 blockNumber;
        address verifier;
        uint256[] sampleIndices;     // Which samples to re-check
        bool resolved;
        bool passed;
    }

    // ── State ──────────────────────────────────────────────────────

    uint256 public currentBlockNumber;
    uint256 public blockInterval = 60;    // Minimum seconds between blocks
    uint256 public lastBlockTime;
    uint256 public difficulty = 1000;     // Minimum samples required per block
    uint256 public verificationQuorum = 3; // Number of verifiers needed

    mapping(uint256 => QuantumBlock) public blocks;
    mapping(uint256 => VerificationChallenge[]) public challenges;
    mapping(uint256 => uint256) public verificationApprovals;
    mapping(uint256 => uint256) public verificationRejections;

    // Pending block submissions (one per interval)
    struct BlockSubmission {
        address miner;
        uint256 problemId;
        bytes32 solutionHash;
        int256 bestEnergy;
        uint256 totalSamples;
        bytes32 sampleMerkleRoot;
        uint256 timestamp;
    }

    BlockSubmission[] public pendingSubmissions;

    event BlockSubmitted(uint256 indexed blockNumber, address indexed miner, int256 energy, uint256 samples);
    event VerificationRequested(uint256 indexed blockNumber, address indexed verifier);
    event VerificationResolved(uint256 indexed blockNumber, bool passed);
    event BlockFinalized(uint256 indexed blockNumber, address indexed miner, uint256 reward);
    event MinerSlashed(uint256 indexed blockNumber, address indexed miner);
    event DifficultyAdjusted(uint256 oldDifficulty, uint256 newDifficulty);

    constructor(address _token, address _staking) {
        token = IQCToken(_token);
        staking = IStaking(_staking);
        lastBlockTime = block.timestamp;
    }

    // ── Mining ─────────────────────────────────────────────────────

    /// @notice Submit a mined block. Miner must be a staked solver.
    function submitBlock(
        uint256 problemId,
        bytes32 solutionHash,
        int256 bestEnergy,
        uint256 totalSamples,
        bytes32 sampleMerkleRoot
    ) external {
        require(staking.isActiveSolver(msg.sender), "Must be active solver");
        require(block.timestamp >= lastBlockTime + blockInterval, "Too soon");
        require(totalSamples >= difficulty, "Below difficulty threshold");

        uint256 blockNum = currentBlockNumber++;
        blocks[blockNum] = QuantumBlock({
            blockNumber: blockNum,
            miner: msg.sender,
            problemId: problemId,
            solutionHash: solutionHash,
            bestEnergy: bestEnergy,
            totalSamples: totalSamples,
            sampleMerkleRoot: sampleMerkleRoot,
            timestamp: block.timestamp,
            verified: false,
            reward: 0
        });

        lastBlockTime = block.timestamp;
        emit BlockSubmitted(blockNum, msg.sender, bestEnergy, totalSamples);

        // Adjust difficulty every 10 blocks
        if (blockNum > 0 && blockNum % 10 == 0) {
            _adjustDifficulty();
        }
    }

    // ── Verification ───────────────────────────────────────────────

    /// @notice Verify a submitted block by spot-checking samples
    function verifyBlock(
        uint256 blockNumber,
        bool approved,
        bytes32 verificationProof  // Hash of the verifier's spot-check results
    ) external {
        require(staking.isActiveVerifier(msg.sender), "Must be active verifier");
        QuantumBlock storage qb = blocks[blockNumber];
        require(!qb.verified, "Already finalized");
        require(qb.miner != msg.sender, "Cannot self-verify");

        // Record verification
        challenges[blockNumber].push(VerificationChallenge({
            blockNumber: blockNumber,
            verifier: msg.sender,
            sampleIndices: new uint256[](0),
            resolved: true,
            passed: approved
        }));

        if (approved) {
            verificationApprovals[blockNumber]++;
        } else {
            verificationRejections[blockNumber]++;
        }

        emit VerificationRequested(blockNumber, msg.sender);

        // Check if quorum reached
        uint256 totalVotes = verificationApprovals[blockNumber] + verificationRejections[blockNumber];
        if (totalVotes >= verificationQuorum) {
            _finalizeBlock(blockNumber);
        }
    }

    function _finalizeBlock(uint256 blockNumber) internal {
        QuantumBlock storage qb = blocks[blockNumber];
        require(!qb.verified, "Already finalized");

        bool passed = verificationApprovals[blockNumber] > verificationRejections[blockNumber];

        if (passed) {
            // Mint reward
            uint256 reward = token.currentMiningReward();
            token.mintMiningReward(qb.miner, reward);
            qb.reward = reward;
            qb.verified = true;

            // Record successful completion
            staking.recordTaskCompletion(qb.miner);

            emit BlockFinalized(blockNumber, qb.miner, reward);
        } else {
            // Slash the miner
            staking.slash(qb.miner, "Block verification failed");
            qb.verified = true;
            qb.reward = 0;

            emit MinerSlashed(blockNumber, qb.miner);
        }

        emit VerificationResolved(blockNumber, passed);
    }

    // ── Difficulty adjustment ──────────────────────────────────────

    function _adjustDifficulty() internal {
        // Target: 1 block per blockInterval seconds
        // If blocks are coming faster, increase difficulty
        // If slower, decrease

        uint256 recentBlocks = 10;
        if (currentBlockNumber < recentBlocks) return;

        uint256 startBlock = currentBlockNumber - recentBlocks;
        uint256 timeSpan = blocks[currentBlockNumber - 1].timestamp - blocks[startBlock].timestamp;
        uint256 targetTime = blockInterval * recentBlocks;

        uint256 oldDifficulty = difficulty;

        if (timeSpan < targetTime * 80 / 100) {
            // Too fast — increase difficulty by 20%
            difficulty = difficulty * 120 / 100;
        } else if (timeSpan > targetTime * 120 / 100) {
            // Too slow — decrease difficulty by 20%
            difficulty = difficulty * 80 / 100;
            if (difficulty < 100) difficulty = 100; // Floor
        }

        if (difficulty != oldDifficulty) {
            emit DifficultyAdjusted(oldDifficulty, difficulty);
        }
    }

    // ── View functions ─────────────────────────────────────────────

    function getBlock(uint256 blockNumber) external view returns (QuantumBlock memory) {
        return blocks[blockNumber];
    }

    function getVerificationStatus(uint256 blockNumber) external view returns (
        uint256 approvals, uint256 rejections, bool finalized
    ) {
        return (
            verificationApprovals[blockNumber],
            verificationRejections[blockNumber],
            blocks[blockNumber].verified
        );
    }
}
