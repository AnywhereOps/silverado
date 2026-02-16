// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

/**
 * @title Rollup — Optimistic Rollup for Quantum Work
 * @notice Quantum simulation results are optimistically accepted and
 *         finalized after a challenge window. Any node can challenge
 *         a submission by requesting Merkle leaf re-verification.
 *
 * Flow:
 *   1. Solver submits Merkle root + best energy + proof hash
 *   2. Immediate provisional acceptance
 *   3. Challenge window opens (5 minutes)
 *   4. If challenged: on-chain dispute resolution
 *   5. If no challenge: finalize, pay solver
 *
 * Key insight: checking 50 random leaves catches a 10% cheater with 99.5%
 * probability. The economics make cheating irrational.
 */
contract Rollup is ReentrancyGuard {

    IERC20 public token;

    uint256 public constant CHALLENGE_WINDOW = 5 minutes;
    uint256 public constant CHALLENGE_DEPOSIT = 100 * 1e18;  // 100 QC to challenge
    uint256 public constant SLASH_REWARD_PCT = 50;  // Challenger gets 50% of solver's slash
    uint256 public constant MAX_CHALLENGED_LEAVES = 100;

    enum SubmissionStatus { Pending, Challenged, Finalized, Rejected }

    struct Submission {
        address solver;
        bytes32 problemHash;
        bytes32 merkleRoot;
        int256 bestEnergy;
        uint256 totalSamples;
        bytes32 zkProofHash;         // Hash of ZK proof (for future ZK verification)
        uint256 timestamp;
        SubmissionStatus status;
        uint256 stakeAmount;         // Solver's stake for this submission
    }

    struct Dispute {
        address challenger;
        uint256 submissionId;
        uint256 challengeDeposit;
        uint256[] leafIndices;       // Which leaves to verify
        bytes32[] revealedLeafHashes; // Solver must reveal these
        bool resolved;
        bool fraudProven;
        uint256 timestamp;
    }

    uint256 public submissionCount;
    mapping(uint256 => Submission) public submissions;
    mapping(uint256 => Dispute[]) public disputes;

    // Settlement tracking
    mapping(bytes32 => uint256) public bestSubmissionForProblem;  // problemHash → submissionId
    mapping(bytes32 => int256) public bestEnergyForProblem;

    event Submitted(uint256 indexed submissionId, address indexed solver, bytes32 merkleRoot, int256 energy);
    event Challenged(uint256 indexed submissionId, address indexed challenger, uint256 disputeIndex);
    event DisputeResolved(uint256 indexed submissionId, uint256 disputeIndex, bool fraudProven);
    event Finalized(uint256 indexed submissionId, address indexed solver, int256 energy);
    event Rejected(uint256 indexed submissionId, address indexed solver, string reason);

    constructor(address _token) {
        token = IERC20(_token);
    }

    /// @notice Submit quantum work results with optimistic acceptance
    function submit(
        bytes32 problemHash,
        bytes32 merkleRoot,
        int256 bestEnergy,
        uint256 totalSamples,
        bytes32 zkProofHash,
        uint256 stakeAmount
    ) external returns (uint256 submissionId) {
        require(totalSamples > 0, "No samples");
        require(stakeAmount > 0, "Must stake tokens");

        // Transfer stake
        token.transferFrom(msg.sender, address(this), stakeAmount);

        submissionId = submissionCount++;
        submissions[submissionId] = Submission({
            solver: msg.sender,
            problemHash: problemHash,
            merkleRoot: merkleRoot,
            bestEnergy: bestEnergy,
            totalSamples: totalSamples,
            zkProofHash: zkProofHash,
            timestamp: block.timestamp,
            status: SubmissionStatus.Pending,
            stakeAmount: stakeAmount
        });

        // Track best solution per problem
        if (bestEnergy > bestEnergyForProblem[problemHash]) {
            bestEnergyForProblem[problemHash] = bestEnergy;
            bestSubmissionForProblem[problemHash] = submissionId;
        }

        emit Submitted(submissionId, msg.sender, merkleRoot, bestEnergy);
    }

    /// @notice Challenge a pending submission
    function challenge(
        uint256 submissionId,
        uint256[] calldata leafIndices
    ) external {
        Submission storage sub = submissions[submissionId];
        require(sub.status == SubmissionStatus.Pending, "Not pending");
        require(block.timestamp < sub.timestamp + CHALLENGE_WINDOW, "Challenge window closed");
        require(leafIndices.length > 0 && leafIndices.length <= MAX_CHALLENGED_LEAVES, "Invalid leaf count");
        require(msg.sender != sub.solver, "Cannot self-challenge");

        // Take challenge deposit
        token.transferFrom(msg.sender, address(this), CHALLENGE_DEPOSIT);

        sub.status = SubmissionStatus.Challenged;

        uint256 disputeIdx = disputes[submissionId].length;
        disputes[submissionId].push(Dispute({
            challenger: msg.sender,
            submissionId: submissionId,
            challengeDeposit: CHALLENGE_DEPOSIT,
            leafIndices: leafIndices,
            revealedLeafHashes: new bytes32[](0),
            resolved: false,
            fraudProven: false,
            timestamp: block.timestamp
        }));

        emit Challenged(submissionId, msg.sender, disputeIdx);
    }

    /// @notice Solver reveals challenged leaves (provides Merkle proofs)
    function revealLeaves(
        uint256 submissionId,
        uint256 disputeIndex,
        bytes32[] calldata leafHashes,
        bytes32[] calldata merkleProof
    ) external {
        Submission storage sub = submissions[submissionId];
        require(msg.sender == sub.solver, "Not solver");

        Dispute storage disp = disputes[submissionId][disputeIndex];
        require(!disp.resolved, "Already resolved");
        require(leafHashes.length == disp.leafIndices.length, "Length mismatch");

        // Store revealed hashes (in production, verify Merkle proofs on-chain)
        for (uint256 i = 0; i < leafHashes.length; i++) {
            disp.revealedLeafHashes.push(leafHashes[i]);
        }
    }

    /// @notice Resolve a dispute (called by authorized verifiers or automatically)
    function resolveDispute(
        uint256 submissionId,
        uint256 disputeIndex,
        bool fraudProven
    ) external {
        Dispute storage disp = disputes[submissionId][disputeIndex];
        require(!disp.resolved, "Already resolved");

        disp.resolved = true;
        disp.fraudProven = fraudProven;

        Submission storage sub = submissions[submissionId];

        if (fraudProven) {
            // Solver loses stake
            uint256 challengerReward = (sub.stakeAmount * SLASH_REWARD_PCT) / 100;
            token.transfer(disp.challenger, disp.challengeDeposit + challengerReward);
            // Rest of stake burned or goes to protocol
            sub.status = SubmissionStatus.Rejected;

            emit Rejected(submissionId, sub.solver, "Fraud proven in dispute");
        } else {
            // Challenge failed — challenger loses deposit
            // Deposit goes to solver as compensation for false accusation
            token.transfer(sub.solver, disp.challengeDeposit);
            // Submission returns to pending (can still be challenged again)
            sub.status = SubmissionStatus.Pending;
        }

        emit DisputeResolved(submissionId, disputeIndex, fraudProven);
    }

    /// @notice Finalize a submission after challenge window expires
    function finalize(uint256 submissionId) external nonReentrant {
        Submission storage sub = submissions[submissionId];
        require(sub.status == SubmissionStatus.Pending, "Not pending");
        require(
            block.timestamp >= sub.timestamp + CHALLENGE_WINDOW,
            "Challenge window not expired"
        );

        sub.status = SubmissionStatus.Finalized;

        // Return solver's stake
        token.transfer(sub.solver, sub.stakeAmount);

        emit Finalized(submissionId, sub.solver, sub.bestEnergy);
    }

    /// @notice ZK verification shortcut — instant finalization with valid ZK proof
    function finalizeWithZKProof(
        uint256 submissionId,
        bytes calldata zkProof
    ) external nonReentrant {
        Submission storage sub = submissions[submissionId];
        require(
            sub.status == SubmissionStatus.Pending ||
            sub.status == SubmissionStatus.Challenged,
            "Cannot finalize"
        );

        // In production: verify the ZK proof on-chain
        // For now: check that proof hash matches
        require(keccak256(zkProof) == sub.zkProofHash, "Invalid ZK proof");

        sub.status = SubmissionStatus.Finalized;
        token.transfer(sub.solver, sub.stakeAmount);

        emit Finalized(submissionId, sub.solver, sub.bestEnergy);
    }

    // ── View functions ─────────────────────────────────────────────

    function getSubmission(uint256 id) external view returns (Submission memory) {
        return submissions[id];
    }

    function getDisputeCount(uint256 submissionId) external view returns (uint256) {
        return disputes[submissionId].length;
    }

    function isChallengeable(uint256 submissionId) external view returns (bool) {
        Submission storage sub = submissions[submissionId];
        return sub.status == SubmissionStatus.Pending &&
               block.timestamp < sub.timestamp + CHALLENGE_WINDOW;
    }

    function isFinalizable(uint256 submissionId) external view returns (bool) {
        Submission storage sub = submissions[submissionId];
        return sub.status == SubmissionStatus.Pending &&
               block.timestamp >= sub.timestamp + CHALLENGE_WINDOW;
    }
}
