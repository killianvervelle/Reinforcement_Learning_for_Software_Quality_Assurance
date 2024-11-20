package net.sourceforge.jFuzzyLogic.membership;

import net.sourceforge.jFuzzyLogic.membership.MembershipFunction;

/**
 * Base continuous membership function
 * @author pcingola@users.sourceforge.net
 */
public abstract class MembershipFunctionContinuous extends MembershipFunction {

	/**
	 * Default constructor 
	 */
	public MembershipFunctionContinuous() {
		super();
		discrete = false;
	}

}
