// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title QCToken — Quantum Carburetor Token
 * @notice ERC-20 token for the distributed quantum simulation network.
 *
 * Used for:
 *   - Problem submission fees (submitters pay solvers)
 *   - Solver staking (slashed if caught cheating)
 *   - Verifier rewards (earn fees for honest verification)
 *   - Block rewards (miners earn for useful quantum computation)
 */

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract QCToken is ERC20, Ownable {
    uint256 public constant MAX_SUPPLY = 1_000_000_000 * 1e18; // 1 billion tokens
    uint256 public totalMinted;

    // Allocation buckets
    uint256 public constant MINING_ALLOCATION = 500_000_000 * 1e18;    // 50% for miners
    uint256 public constant TEAM_ALLOCATION = 150_000_000 * 1e18;      // 15% for team
    uint256 public constant ECOSYSTEM_ALLOCATION = 200_000_000 * 1e18; // 20% for ecosystem
    uint256 public constant STAKING_ALLOCATION = 150_000_000 * 1e18;   // 15% for staking rewards

    uint256 public miningMinted;
    uint256 public stakingMinted;

    // Authorized minters (PoQW contract, staking contract)
    mapping(address => bool) public authorizedMinters;

    event MinterAuthorized(address indexed minter);
    event MinterRevoked(address indexed minter);

    constructor(address teamWallet, address ecosystemWallet)
        ERC20("Quantum Carburetor", "QC")
        Ownable(msg.sender)
    {
        // Mint team and ecosystem allocations upfront
        _mint(teamWallet, TEAM_ALLOCATION);
        _mint(ecosystemWallet, ECOSYSTEM_ALLOCATION);
        totalMinted = TEAM_ALLOCATION + ECOSYSTEM_ALLOCATION;
    }

    modifier onlyAuthorizedMinter() {
        require(authorizedMinters[msg.sender], "Not authorized minter");
        _;
    }

    function authorizeMinter(address minter) external onlyOwner {
        authorizedMinters[minter] = true;
        emit MinterAuthorized(minter);
    }

    function revokeMinter(address minter) external onlyOwner {
        authorizedMinters[minter] = false;
        emit MinterRevoked(minter);
    }

    /// @notice Mint mining rewards. Called by PoQW contract.
    function mintMiningReward(address to, uint256 amount) external onlyAuthorizedMinter {
        require(miningMinted + amount <= MINING_ALLOCATION, "Mining allocation exhausted");
        miningMinted += amount;
        totalMinted += amount;
        _mint(to, amount);
    }

    /// @notice Mint staking rewards. Called by Staking contract.
    function mintStakingReward(address to, uint256 amount) external onlyAuthorizedMinter {
        require(stakingMinted + amount <= STAKING_ALLOCATION, "Staking allocation exhausted");
        stakingMinted += amount;
        totalMinted += amount;
        _mint(to, amount);
    }

    /// @notice Current mining reward per block (halving schedule)
    function currentMiningReward() public view returns (uint256) {
        // Simple halving every 25% of mining allocation
        if (miningMinted < MINING_ALLOCATION / 4) {
            return 100 * 1e18; // 100 QC per block
        } else if (miningMinted < MINING_ALLOCATION / 2) {
            return 50 * 1e18;
        } else if (miningMinted < (MINING_ALLOCATION * 3) / 4) {
            return 25 * 1e18;
        } else {
            return 12.5e18;   // Use fixed point: 12.5 QC
        }
    }
}
