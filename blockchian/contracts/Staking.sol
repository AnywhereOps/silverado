// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title Staking — Solver and Verifier stake management
 * @notice Solvers stake QC tokens to participate in the network.
 *         Stake is slashed if caught cheating. Verifiers also stake
 *         to prove they have skin in the game.
 *
 * Reputation builds over time:
 *   - Successful verifications increase reputation
 *   - Caught cheating slashes stake AND reputation
 *   - Higher reputation = priority in task assignment
 */
contract Staking is ReentrancyGuard {

    IERC20 public token;

    uint256 public constant MIN_SOLVER_STAKE = 1000 * 1e18;   // 1,000 QC
    uint256 public constant MIN_VERIFIER_STAKE = 500 * 1e18;   // 500 QC
    uint256 public constant SLASH_PERCENTAGE = 50;              // 50% slash on cheating
    uint256 public constant COOLDOWN_PERIOD = 7 days;

    enum Role { None, Solver, Verifier }

    struct StakeInfo {
        uint256 amount;
        Role role;
        uint256 reputation;          // 0-10000 (basis points, 10000 = 100%)
        uint256 tasksCompleted;
        uint256 tasksFailed;
        uint256 slashCount;
        uint256 lastStakeTime;
        uint256 unstakeRequestTime;  // 0 if not requested
        bool active;
    }

    mapping(address => StakeInfo) public stakes;
    address[] public solvers;
    address[] public verifiers;

    // Authorized slashers (PoQW contract, verification contract)
    mapping(address => bool) public authorizedSlashers;
    address public owner;

    event Staked(address indexed staker, Role role, uint256 amount);
    event UnstakeRequested(address indexed staker, uint256 availableAt);
    event Unstaked(address indexed staker, uint256 amount);
    event Slashed(address indexed staker, uint256 amount, string reason);
    event ReputationUpdated(address indexed staker, uint256 newReputation);
    event TaskCompleted(address indexed staker);

    constructor(address _token) {
        token = IERC20(_token);
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier onlyAuthorizedSlasher() {
        require(authorizedSlashers[msg.sender], "Not authorized slasher");
        _;
    }

    function authorizeSlasher(address slasher) external onlyOwner {
        authorizedSlashers[slasher] = true;
    }

    /// @notice Stake tokens as a solver or verifier
    function stake(Role role, uint256 amount) external nonReentrant {
        require(role == Role.Solver || role == Role.Verifier, "Invalid role");
        require(!stakes[msg.sender].active, "Already staked");

        uint256 minStake = role == Role.Solver ? MIN_SOLVER_STAKE : MIN_VERIFIER_STAKE;
        require(amount >= minStake, "Below minimum stake");

        token.transferFrom(msg.sender, address(this), amount);

        stakes[msg.sender] = StakeInfo({
            amount: amount,
            role: role,
            reputation: 5000,  // Start at 50%
            tasksCompleted: 0,
            tasksFailed: 0,
            slashCount: 0,
            lastStakeTime: block.timestamp,
            unstakeRequestTime: 0,
            active: true
        });

        if (role == Role.Solver) {
            solvers.push(msg.sender);
        } else {
            verifiers.push(msg.sender);
        }

        emit Staked(msg.sender, role, amount);
    }

    /// @notice Request unstaking (begins cooldown period)
    function requestUnstake() external {
        StakeInfo storage info = stakes[msg.sender];
        require(info.active, "Not staked");
        require(info.unstakeRequestTime == 0, "Already requested");

        info.unstakeRequestTime = block.timestamp;
        emit UnstakeRequested(msg.sender, block.timestamp + COOLDOWN_PERIOD);
    }

    /// @notice Complete unstaking after cooldown
    function unstake() external nonReentrant {
        StakeInfo storage info = stakes[msg.sender];
        require(info.active, "Not staked");
        require(info.unstakeRequestTime > 0, "Unstake not requested");
        require(
            block.timestamp >= info.unstakeRequestTime + COOLDOWN_PERIOD,
            "Cooldown not complete"
        );

        uint256 amount = info.amount;
        info.active = false;
        info.amount = 0;
        info.unstakeRequestTime = 0;

        token.transfer(msg.sender, amount);
        emit Unstaked(msg.sender, amount);
    }

    /// @notice Slash a staker for cheating
    function slash(address staker, string calldata reason) external onlyAuthorizedSlasher {
        StakeInfo storage info = stakes[staker];
        require(info.active, "Not staked");

        uint256 slashAmount = (info.amount * SLASH_PERCENTAGE) / 100;
        info.amount -= slashAmount;
        info.slashCount++;

        // Reputation hit: -2000 per slash (20%)
        if (info.reputation > 2000) {
            info.reputation -= 2000;
        } else {
            info.reputation = 0;
        }

        // Slashed tokens go to the fee collector / burned
        // For now, just stay in contract
        emit Slashed(staker, slashAmount, reason);
        emit ReputationUpdated(staker, info.reputation);

        // Auto-deactivate if stake drops below minimum
        uint256 minStake = info.role == Role.Solver ? MIN_SOLVER_STAKE : MIN_VERIFIER_STAKE;
        if (info.amount < minStake) {
            info.active = false;
            // Return remaining stake
            if (info.amount > 0) {
                uint256 remaining = info.amount;
                info.amount = 0;
                token.transfer(staker, remaining);
            }
        }
    }

    /// @notice Record successful task completion (increases reputation)
    function recordTaskCompletion(address staker) external onlyAuthorizedSlasher {
        StakeInfo storage info = stakes[staker];
        require(info.active, "Not staked");

        info.tasksCompleted++;

        // Reputation increase: +10 per task, max 10000
        if (info.reputation < 10000) {
            info.reputation = info.reputation + 10 > 10000 ? 10000 : info.reputation + 10;
        }

        emit TaskCompleted(staker);
        emit ReputationUpdated(staker, info.reputation);
    }

    /// @notice Record task failure (decreases reputation but doesn't slash)
    function recordTaskFailure(address staker) external onlyAuthorizedSlasher {
        StakeInfo storage info = stakes[staker];
        require(info.active, "Not staked");

        info.tasksFailed++;

        // Small reputation decrease: -50 per failure
        if (info.reputation > 50) {
            info.reputation -= 50;
        } else {
            info.reputation = 0;
        }

        emit ReputationUpdated(staker, info.reputation);
    }

    // ── View functions ─────────────────────────────────────────────

    function getStakeInfo(address staker) external view returns (StakeInfo memory) {
        return stakes[staker];
    }

    function isActiveSolver(address addr) external view returns (bool) {
        StakeInfo storage info = stakes[addr];
        return info.active && info.role == Role.Solver;
    }

    function isActiveVerifier(address addr) external view returns (bool) {
        StakeInfo storage info = stakes[addr];
        return info.active && info.role == Role.Verifier;
    }

    function getSolverCount() external view returns (uint256) {
        return solvers.length;
    }

    function getVerifierCount() external view returns (uint256) {
        return verifiers.length;
    }
}
